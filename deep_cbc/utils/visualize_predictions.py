import argparse
import logging
import shutil
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from omegaconf import OmegaConf
from PIL import Image
from PIL import ImageDraw as D
from torchvision import transforms
from tqdm import tqdm
from util.func import get_patch_size
from util.vis_pipnet import get_img_coordinates

logger = logging.getLogger()
logger.setLevel(logging.INFO)

try:
    import cv2

    use_opencv = True
except ImportError:
    use_opencv = False
    logging.info(
        "Heatmaps showing where a prototype is found will not be generated because OpenCV is not installed."
    )


def visualize_predictions(
    net, vis_test_dir, classes, device, args: Union[argparse.Namespace, OmegaConf]
):

    # Make sure the model is in evaluation mode
    net.eval()

    save_dir = Path(args.log_dir) / Path(args.dir_for_saving_images)
    if save_dir.exists():
        shutil.rmtree(str(save_dir))

    patchsize, skip = get_patch_size(args)

    num_workers = args.num_workers

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    normalize = transforms.Normalize(mean=mean, std=std)
    transform_no_augment = transforms.Compose(
        [
            transforms.Resize(size=(args.image_size, args.image_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )

    vis_test_set = torchvision.datasets.ImageFolder(
        vis_test_dir, transform=transform_no_augment
    )
    vis_test_loader = torch.utils.data.DataLoader(
        vis_test_set,
        batch_size=1,
        shuffle=False,
        pin_memory=not args.disable_cuda and torch.cuda.is_available(),
        num_workers=num_workers,
    )
    imgs = vis_test_set.imgs

    last_y = -1
    for k, (xs, ys) in tqdm(
        enumerate(vis_test_loader)
    ):  # shuffle is false so should lead to same order as in imgs
        if ys[0] != last_y:
            last_y = ys[0]
            count_per_y = 0
        else:
            count_per_y += 1
            if count_per_y > 5:  # show max 5 imgs per class to speed up the process
                continue
        xs, ys = xs.to(device), ys.to(device)
        img = imgs[k][0]

        img_name = Path(img).stem
        dir = save_dir / Path(img_name)
        if not dir.exists():
            dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(str(img), str(dir))

        with torch.no_grad():
            # "softmaxes" has Shape: (bs, num_prototypes, W, H),
            # "pooled" has Shape: (bs, num_prototypes),
            # "out" has Shape: (bs, num_classes).
            softmaxes, pooled, out = net(xs, inference=True)
            sorted_out, sorted_out_indices = torch.sort(out.squeeze(0), descending=True)
            for pred_class_idx in sorted_out_indices[:3]:
                pred_class = classes[pred_class_idx]
                pred_class_dir = Path(
                    pred_class + "_" + str(f"{out[0, pred_class_idx].item():.5f}")
                )
                save_path = dir / pred_class_dir
                if not save_path.exists():
                    save_path.mkdir(parents=True, exist_ok=True)
                sorted_pooled, sorted_pooled_indices = torch.sort(
                    pooled.squeeze(0), descending=True
                )
                simweights = []
                for prototype_idx in sorted_pooled_indices:

                    # Note: The "rbf_head" have the same implementation as the "pipnet_head" but instead of final ReLU layer a Softmax layer is used.
                    if args.head_type in ("pipnet_head", "rbf_head"):
                        classification_weights = net.module._classification.weight
                    elif args.head_type in ("cbc_head"):
                        if args.reasoning_type == "positive":
                            classification_weights = net.module._classification.effective_reasoning_probabilities[
                                0
                            ]  # [0]: for positive reasoning, [1]: for negative reasoning from  effective_reasoning_probabilities
                        elif args.reasoning_type == "negative":
                            classification_weights = net.module._classification.effective_reasoning_probabilities[
                                1
                            ]  # [0]: for positive reasoning, [1]: for negative reasoning
                        classification_weights = torch.transpose(
                            classification_weights, 0, 1
                        )

                    simweight = (
                        pooled[0, prototype_idx].item()
                        * classification_weights[pred_class_idx, prototype_idx].item()
                    )
                    simweights.append(simweight)

                    # The threshold to ignore prototypes that are not relevant to any class.
                    # The below c_weight_threshold set based on empirical runs for faster execution
                    # by obtaining relevant enough amount of prototypes.
                    simweight_threshold = 1e-5
                    if args.head_type == "cbc_head":
                        simweight_threshold = 7.5e-4
                    if (
                        args.head_type == "cbc_head"
                        and "resnet" in args.net
                        and args.reasoning_type == "negative"
                    ):
                        simweight_threshold = 4.5e-4
                    if simweight_threshold > simweight_threshold:

                        max_h, max_idx_h = torch.max(
                            softmaxes[0, prototype_idx, :, :], dim=0
                        )
                        max_w, max_idx_w = torch.max(max_h, dim=0)
                        max_idx_h = max_idx_h[max_idx_w].item()
                        max_idx_w = max_idx_w.item()
                        image = transforms.Resize(
                            size=(args.image_size, args.image_size)
                        )(Image.open(img).convert("RGB"))
                        img_tensor = transforms.ToTensor()(image).unsqueeze_(
                            0
                        )  # shape (1, 3, h, w)
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
                            max_idx_h,
                            max_idx_w,
                        )
                        img_tensor_patch = img_tensor[
                            0, :, h_coor_min:h_coor_max, w_coor_min:w_coor_max
                        ]
                        img_patch = transforms.ToPILImage()(img_tensor_patch)

                        img_patch_file_name = "mul{}_p{}_sim{}_w{}_patch.png".format(
                            str(f"{simweight:.5f}"),
                            str(prototype_idx.item()),
                            str(f"{pooled[0, prototype_idx].item():.5f}"),
                            str(
                                f"{classification_weights[pred_class_idx, prototype_idx].item():.5f}"
                            ),
                        )
                        img_patch.save(save_path / Path(img_patch_file_name))

                        draw = D.Draw(image)
                        draw.rectangle(
                            [
                                (max_idx_w * skip, max_idx_h * skip),
                                (
                                    min(args.image_size, max_idx_w * skip + patchsize),
                                    min(args.image_size, max_idx_h * skip + patchsize),
                                ),
                            ],
                            outline="yellow",
                            width=2,
                        )

                        img_rect_file_name = "mul{}_p{}_sim{}_w{}_rect.png".format(
                            str(f"{simweight:.5f}"),
                            str(prototype_idx.item()),
                            str(f"{pooled[0, prototype_idx].item():.5f}"),
                            str(
                                f"{classification_weights[pred_class_idx, prototype_idx].item():.5f}"
                            ),
                        )

                        image.save(save_path / Path(img_rect_file_name))

                        # Visualise softmax as heatmap and for switching it off, activate the below flag.
                        # if False:
                        if (
                            use_opencv and args.dataset != "CARS"
                        ):  # to remove the error for the CARS dataset.
                            softmaxes_resized = transforms.ToPILImage()(
                                softmaxes[0, prototype_idx, :, :]
                            )
                            softmaxes_resized = softmaxes_resized.resize(
                                (args.image_size, args.image_size), Image.BICUBIC
                            )
                            softmaxes_np = (
                                (transforms.ToTensor()(softmaxes_resized))
                                .squeeze()
                                .numpy()
                            )

                            heatmap = cv2.applyColorMap(
                                np.uint8(255 * softmaxes_np), cv2.COLORMAP_JET
                            )
                            heatmap = np.float32(heatmap) / 255
                            heatmap = heatmap[..., ::-1]  # OpenCV's BGR to RGB
                            heatmap_img = 0.2 * np.float32(heatmap) + 0.6 * np.float32(
                                img_tensor.squeeze().numpy().transpose(1, 2, 0)
                            )
                            image_heatmap_file_name = "heatmap_p%s.png" % str(
                                prototype_idx.item()
                            )
                            plt.imsave(
                                fname=save_path / image_heatmap_file_name,
                                arr=heatmap_img,
                                vmin=0.0,
                                vmax=1.0,
                            )
