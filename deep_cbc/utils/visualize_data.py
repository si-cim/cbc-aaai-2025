import argparse
import logging
import random
from pathlib import Path
from typing import Union

import torch
import torch.nn.functional as F
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from omegaconf import OmegaConf

# import os
from PIL import Image
from PIL import ImageDraw as D
from tqdm import tqdm
from util.func import get_patch_size

logger = logging.getLogger()
logger.setLevel(logging.INFO)


@torch.no_grad()
def visualize_top_k(
    net,
    projectloader,
    num_classes,
    device,
    foldername,
    args: Union[argparse.Namespace, OmegaConf],
    k=10,
):
    logging.info("Visualizing prototypes for top-k...")
    dir = Path(args.log_dir) / Path(foldername)
    if not dir.exists():
        dir.mkdir(parents=True, exist_ok=True)

    near_imgs_dirs = dict()
    seen_max = dict()
    saved = dict()
    saved_ys = dict()
    tensors_per_prototype = dict()

    for p in range(net.module._num_prototypes):
        near_imgs_dir = dir / Path(str(p))
        near_imgs_dirs[p] = near_imgs_dir
        seen_max[p] = 0.0
        saved[p] = 0
        saved_ys[p] = []
        tensors_per_prototype[p] = []

    patchsize, skip = get_patch_size(args)

    imgs = projectloader.dataset.imgs

    # Make sure the model is in evaluation mode.
    net.eval()

    # Note: The "rbf_head" have the same implementation as the "pipnet_head" but instead of final ReLU layer a Softmax layer is used.
    if args.head_type in ("pipnet_head", "rbf_head"):
        classification_weights = net.module._classification.weight
    elif args.head_type in ("cbc_head"):
        if args.reasoning_type == "positive":
            classification_weights = net.module._classification.effective_reasoning_probabilities[
                0
            ]  # [0]: for positive reasoning, [1]: for negative reasoning from  effective_reasoning_probabilities
        elif args.reasoning_type == "negative":
            classification_weights = (
                net.module._classification.effective_reasoning_probabilities[1]
            )  # [0]: for positive reasoning, [1]: for negative reasoning
        classification_weights = torch.transpose(classification_weights, 0, 1)

    # Show progress on progress bar.
    img_iter = tqdm(
        enumerate(projectloader),
        total=len(projectloader),
        mininterval=50.0,
        desc="Collecting top-k...",
        ncols=0,
    )

    # Iterate through the data.
    images_seen = 0
    topks = dict()
    # Iterate through the training set.
    for i, (xs, ys) in img_iter:
        images_seen += 1
        xs, ys = xs.to(device), ys.to(device)

        with torch.no_grad():
            # Use the model to classify this batch of input data.
            pfs, pooled, _ = net(
                xs, inference=True
            )  # inference=True is only going to get triggered for head_type="pipnet_head".
            pooled = pooled.squeeze(0)
            pfs = pfs.squeeze(0)
            for p in range(pooled.shape[0]):
                c_weight = torch.max(classification_weights[:, p])
                # The threshold to ignore prototypes that are not relevant to any class.
                # The below c_weight_threshold set based on empirical runs for faster execution
                # by obtaining relevant enough amount of prototypes.
                c_weight_threshold = 1e-5
                if args.head_type == "cbc_head":
                    c_weight_threshold = 7.5e-4
                if (
                    args.head_type == "cbc_head"
                    and "resnet" in args.net
                    and args.reasoning_type == "negative"
                ):
                    c_weight_threshold = 4.5e-4
                if c_weight > c_weight_threshold:
                    if p not in topks.keys():
                        topks[p] = []

                    if len(topks[p]) < k:
                        topks[p].append((i, pooled[p].item()))
                    else:
                        topks[p] = sorted(
                            topks[p], key=lambda tup: tup[1], reverse=True
                        )

                        if topks[p][-1][1] < pooled[p].item():
                            topks[p][-1] = (i, pooled[p].item())
                        if topks[p][-1][1] == pooled[p].item():
                            # Equal scores: randomly chose one (since dataset is not shuffled so latter images with same scores can now also get in top-k).
                            replace_choice = random.choice([0, 1])
                            if replace_choice > 0:
                                topks[p][-1] = (i, pooled[p].item())

    alli = []
    prototypes_not_used = []
    for p in topks.keys():
        found = False
        for idx, score in topks[p]:
            alli.append(idx)
            if (
                score > 0.1
            ):  # In case prototypes have fewer than k well-related patches.
                found = True
        if not found:
            prototypes_not_used.append(p)

    logging.info(
        "%s prototypes do not have any similarity score > 0.1. Will be ignored in visualisation.",
        str(len(prototypes_not_used)),
    )

    abstained = 0
    # Show progress on progress bar.
    img_iter = tqdm(
        enumerate(projectloader),
        total=len(projectloader),
        mininterval=50.0,
        desc="Visualizing top-k...",
        ncols=0,
    )
    for i, (
        xs,
        ys,
    ) in img_iter:  # Shuffle is false so should lead to same order as in imgs.
        if i in alli:
            xs, ys = xs.to(device), ys.to(device)
            for p in topks.keys():
                if p not in prototypes_not_used:
                    for idx, score in topks[p]:
                        if idx == i:
                            # Use the model to classify this batch of input data.
                            with torch.no_grad():
                                softmaxes, pooled, out = net(
                                    xs, inference=True
                                )  # Shape: Softmax values have shape of (1, num_prototypes, W, H).
                                outmax = torch.amax(out, dim=1)[
                                    0
                                ]  # Shape: ([1]) because batch size of project loader is 1.
                                if outmax.item() == 0.0:
                                    abstained += 1

                            # Take the max per prototype.
                            max_per_prototype, max_idx_per_prototype = torch.max(
                                softmaxes, dim=0
                            )
                            max_per_prototype_h, max_idx_per_prototype_h = torch.max(
                                max_per_prototype, dim=1
                            )
                            max_per_prototype_w, max_idx_per_prototype_w = torch.max(
                                max_per_prototype_h, dim=1
                            )  # Shape: (num_prototypes).

                            c_weight = torch.max(
                                classification_weights[:, p]
                            )  # Ignore prototypes that are not relevant to any class.
                            if (c_weight > 1e-10) or ("pretrain" in foldername):

                                h_idx = max_idx_per_prototype_h[
                                    p, max_idx_per_prototype_w[p]
                                ]
                                w_idx = max_idx_per_prototype_w[p]

                                img_to_open = imgs[i]
                                if isinstance(img_to_open, tuple) or isinstance(
                                    img_to_open, list
                                ):  # Dataset contains tuples of (img,label).
                                    img_to_open = img_to_open[0]

                                image = transforms.Resize(
                                    size=(args.image_size, args.image_size)
                                )(Image.open(img_to_open).convert("RGB"))
                                img_tensor = transforms.ToTensor()(image).unsqueeze_(
                                    0
                                )  # Shape: (1, 3, h, w)
                                (
                                    h_coor_min,
                                    h_coor_max,
                                    w_coor_min,
                                    w_coor_max,
                                ) = get_img_coordinates(
                                    args.image_size,
                                    softmaxes.shape,
                                    patchsize,
                                    skip,
                                    h_idx,
                                    w_idx,
                                )
                                img_tensor_patch = img_tensor[
                                    0, :, h_coor_min:h_coor_max, w_coor_min:w_coor_max
                                ]

                                saved[p] += 1
                                tensors_per_prototype[p].append(img_tensor_patch)

    logging.info("Abstained: " + str(abstained))
    all_tensors = []
    for p in range(net.module._num_prototypes):
        if saved[p] > 0:
            # Add text next to each topk-grid, to easily see which prototype it is.
            text = "P " + str(p)
            txtimage = Image.new(
                "RGB", (img_tensor_patch.shape[1], img_tensor_patch.shape[2]), (0, 0, 0)
            )
            draw = D.Draw(txtimage)
            draw.text(
                (img_tensor_patch.shape[0] // 2, img_tensor_patch.shape[1] // 2),
                text,
                anchor="mm",
                fill="white",
            )
            txttensor = transforms.ToTensor()(txtimage)
            tensors_per_prototype[p].append(txttensor)
            # Save top-k image patches in grid.
            try:
                grid = torchvision.utils.make_grid(
                    tensors_per_prototype[p], nrow=k + 1, padding=1
                )
                grid_file_path = Path("grid_topk_%s.png" % (str(p)))
                torchvision.utils.save_image(grid, dir / grid_file_path)
                if saved[p] >= k:
                    all_tensors += tensors_per_prototype[p]
            except:
                pass
    if len(all_tensors) > 0:
        grid = torchvision.utils.make_grid(all_tensors, nrow=k + 1, padding=1)
        torchvision.utils.save_image(grid, dir / Path("grid_topk_all.png"))
    else:
        logging.info("Pretrained prototypes not visualized. Try to pretrain longer.")
    return topks


def visualize_data(
    net,
    projectloader,
    num_classes,
    device,
    foldername,
    args: Union[argparse.Namespace, OmegaConf],
):
    logging.info("Visualizing prototypes...")
    dir = Path(args.log_dir) / Path(foldername)
    if not dir.exists():
        dir.mkdir(parents=True, exist_ok=True)

    near_imgs_dirs = dict()
    seen_max = dict()
    saved = dict()
    saved_ys = dict()
    tensors_per_prototype = dict()
    abstainedimgs = set()
    notabstainedimgs = set()

    for p in range(net.module._num_prototypes):
        near_imgs_dir = dir / Path(str(p))
        near_imgs_dirs[p] = near_imgs_dir
        seen_max[p] = 0.0
        saved[p] = 0
        saved_ys[p] = []
        tensors_per_prototype[p] = []

    patchsize, skip = get_patch_size(args)

    imgs = projectloader.dataset.imgs

    # Skip some images for visualisation to speed up the process.
    if len(imgs) / num_classes < 10:
        skip_img = 10
    elif len(imgs) / num_classes < 50:
        skip_img = 5
    else:
        skip_img = 2
    logging.info(
        "Every %s is skipped in order to speed up the visualisation process.",
        str(skip_img),
    )

    # Make sure the model is in evaluation mode.
    net.eval()

    # Note: The "rbf_head" have the same implementation as the "pipnet_head".
    # But, instead of final ReLU layer a Softmax layer is used.
    if args.head_type in ("pipnet_head", "rbf_head"):
        classification_weights = net.module._classification.weight
    elif args.head_type in ("cbc_head"):
        if args.reasoning_type == "positive":
            # Selection with [0]: for positive reasoning, [1]: for negative reasoning
            # from effective_reasoning_probabilities.
            classification_weights = (
                net.module._classification.effective_reasoning_probabilities[0]
            )
        elif args.reasoning_type == "negative":
            classification_weights = (
                net.module._classification.effective_reasoning_probabilities[1]
            )
        classification_weights = torch.transpose(classification_weights, 0, 1)

    # Show progress on progress bar
    img_iter = tqdm(
        enumerate(projectloader),
        total=len(projectloader),
        mininterval=100.0,
        desc="Visualizing...",
        ncols=0,
    )

    # Iterate through the data
    images_seen_before = 0
    for i, (
        xs,
        ys,
    ) in img_iter:  # shuffle is false so should lead to same order as in imgs
        if i % skip_img == 0:
            images_seen_before += xs.shape[0]
            continue

        xs, ys = xs.to(device), ys.to(device)
        # Use the model to classify this batch of input data
        with torch.no_grad():
            softmaxes, _, out = net(xs, inference=True)
            # if args.reasoning_type == "negative":
            #     softmaxes = 1 - softmaxes

        max_per_prototype, max_idx_per_prototype = torch.max(softmaxes, dim=0)
        # In PyTorch, images are represented as [channels, height, width]
        max_per_prototype_h, max_idx_per_prototype_h = torch.max(
            max_per_prototype, dim=1
        )
        max_per_prototype_w, max_idx_per_prototype_w = torch.max(
            max_per_prototype_h, dim=1
        )
        for p in range(0, net.module._num_prototypes):
            c_weight = torch.max(
                classification_weights[:, p]
            )  # ignore prototypes that are not relevant to any class
            if c_weight > 0:
                h_idx = max_idx_per_prototype_h[p, max_idx_per_prototype_w[p]]
                w_idx = max_idx_per_prototype_w[p]
                idx_to_select = max_idx_per_prototype[p, h_idx, w_idx].item()
                found_max = max_per_prototype[p, h_idx, w_idx].item()

                imgname = imgs[images_seen_before + idx_to_select]
                if out.max() < 1e-8:
                    abstainedimgs.add(imgname)
                else:
                    notabstainedimgs.add(imgname)

                if found_max > seen_max[p]:
                    seen_max[p] = found_max

                if found_max > 0.5:
                    img_to_open = imgs[images_seen_before + idx_to_select]
                    if isinstance(img_to_open, tuple) or isinstance(
                        img_to_open, list
                    ):  # dataset contains tuples of (img,label)
                        imglabel = img_to_open[1]
                        img_to_open = img_to_open[0]

                    image = transforms.Resize(size=(args.image_size, args.image_size))(
                        Image.open(img_to_open).convert("RGB")
                    )
                    img_tensor = transforms.ToTensor()(image).unsqueeze_(
                        0
                    )  # shape (1, 3, h, w)
                    (
                        h_coor_min,
                        h_coor_max,
                        w_coor_min,
                        w_coor_max,
                    ) = get_img_coordinates(
                        args.image_size, softmaxes.shape, patchsize, skip, h_idx, w_idx
                    )
                    img_tensor_patch = img_tensor[
                        0, :, h_coor_min:h_coor_max, w_coor_min:w_coor_max
                    ]
                    saved[p] += 1
                    tensors_per_prototype[p].append((img_tensor_patch, found_max))

                    prototype_file_path = Path("prototype_" + str(p))
                    save_path = dir / prototype_file_path
                    if not save_path.exists():
                        save_path.mkdir(parents=True, exist_ok=True)
                    draw = D.Draw(image)
                    draw.rectangle(
                        [(w_coor_min, h_coor_min), (w_coor_max, h_coor_max)],
                        outline="yellow",
                        width=2,
                    )
                    image_rect_coord_file_name = "p{}_{}_{}_{}_rect.png".format(
                        str(p),
                        str(imglabel),
                        str(round(found_max, 2)),
                        str(img_to_open.split("/")[-1].split(".jpg")[0]),
                    )
                    image.save(save_path / Path(image_rect_coord_file_name))

        images_seen_before += len(ys)

    logging.info("num images abstained: %s", str(len(abstainedimgs)))
    logging.info("num images not abstained: %s", str(len(notabstainedimgs)))
    for p in range(net.module._num_prototypes):
        if saved[p] > 0:
            try:
                sorted_by_second = sorted(
                    tensors_per_prototype[p], key=lambda tup: tup[1], reverse=True
                )
                sorted_ps = [i[0] for i in sorted_by_second]
                grid = torchvision.utils.make_grid(sorted_ps, nrow=16, padding=1)
                grid_file_path = Path("grid_topk_%s.png" % (str(p)))
                torchvision.utils.save_image(grid, dir / grid_file_path)
            except RuntimeError:
                pass


# convert latent location to coordinates of image patch
def get_img_coordinates(img_size, softmaxes_shape, patchsize, skip, h_idx, w_idx):
    # in case latent output size is 26x26. For convnext with smaller strides.
    if softmaxes_shape[1] == 26 and softmaxes_shape[2] == 26:
        # Since the outer latent patches have a smaller receptive field, skip size is set to 4 for the first and last patch. 8 for rest.
        h_coor_min = max(0, (h_idx - 1) * skip + 4)
        if h_idx < softmaxes_shape[-1] - 1:
            h_coor_max = h_coor_min + patchsize
        else:
            h_coor_min -= 4
            h_coor_max = h_coor_min + patchsize
        w_coor_min = max(0, (w_idx - 1) * skip + 4)
        if w_idx < softmaxes_shape[-1] - 1:
            w_coor_max = w_coor_min + patchsize
        else:
            w_coor_min -= 4
            w_coor_max = w_coor_min + patchsize
    else:
        h_coor_min = h_idx * skip
        h_coor_max = min(img_size, h_idx * skip + patchsize)
        w_coor_min = w_idx * skip
        w_coor_max = min(img_size, w_idx * skip + patchsize)

    if h_idx == softmaxes_shape[1] - 1:
        h_coor_max = img_size
    if w_idx == softmaxes_shape[2] - 1:
        w_coor_max = img_size
    if h_coor_max == img_size:
        h_coor_min = img_size - patchsize
    if w_coor_max == img_size:
        w_coor_min = img_size - patchsize

    return h_coor_min, h_coor_max, w_coor_min, w_coor_max
