import argparse
import random
from typing import Union

import numpy as np
import torch
import torch.optim
from omegaconf import OmegaConf


def get_patch_size(args):
    patchsize = 32
    skip = round((args.image_size - patchsize) / (args.wshape - 1))
    return patchsize, skip


def init_weights_xavier(m):
    if type(m) == torch.nn.Conv2d:
        torch.nn.init.xavier_uniform_(
            m.weight, gain=torch.nn.init.calculate_gain("sigmoid")
        )


# https://gist.github.com/weiaicunzai/2a5ae6eac6712c70bde0630f3e76b77b?permalink_comment_id=3662215#gistcomment-3662215
def topk_accuracy(
    output,
    target,
    topk=[
        1,
    ],
):
    """
    Computes the accuracy over the k top predictions for the specified values of k
    """
    with torch.no_grad():
        topk2 = [
            x for x in topk if x <= output.shape[1]
        ]  # ensures that k is not larger than number of classes
        maxk = max(topk2)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = (pred == target.unsqueeze(dim=0)).expand_as(pred)

        res = []
        for k in topk:
            if k in topk2:
                correct_k = correct[:k].reshape(-1).float()
                res.append(correct_k)
            else:
                res.append(torch.zeros_like(target))
        return res


def get_optimizers_nn(
    net, args: Union[argparse.Namespace, OmegaConf], optimizer_type: str = "complete"
) -> torch.optim.Optimizer:
    """
    Based on optimizer_type "separate" or "complete" it returns below stated two or three optimizers for model training.

    1. This optimizer returns two different optimizers for training the backbone and the classification head respectively.
        Note: It is important to note that there are two separate optimizers that optimize the network in this
        implementation and this implementation is used for training PIPNet and RBFNet. So, actual end-to-end claim is
        not entirely correct when model training is carried for network architectures like PIPNet.

    2. This optimizer returns three different optimizers for training the backbone-only, the complete end-to-end
    training and the classification head separately. We use end-to-end optimizer to train our CBCNet and
    backbone-only optimizer is only for pre-training step not anywhere in the classification training step.

    :param net: The torch model selected for setting the corresponding gradients to be trained.
    :param args: The training arguments or experiment config to train the prototype based network.
    :param optimizer_type: The optimizer type selected for training different types of prototype based networks.
    :return: Returns separate torch optimizers and parameters of the model.
    """
    assert optimizer_type in (
        "complete",
        "separate",
    ), "Incorrect optimizer value is specified, refer documentation."

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # create parameter groups
    params_to_freeze = []
    params_to_train = []
    params_backbone = []
    # set up optimizer
    if "resnet50" in args.net:
        # freeze resnet50 except last convolutional layer
        for name, param in net.module._net.named_parameters():
            if "layer4.2" in name:
                params_to_train.append(param)
            elif "layer4" in name or "layer3" in name:
                params_to_freeze.append(
                    param
                )  # Note: These parameters are not frozen but rather the backbone params below are frozen.
            elif "layer2" in name:
                params_backbone.append(param)
            else:  # such that model training fits on one gpu.
                param.requires_grad = False
                # params_backbone.append(param)
    elif "convnext" in args.net:
        print("chosen network is convnext", flush=True)
        for name, param in net.module._net.named_parameters():
            if "features.7.2" in name:
                params_to_train.append(param)
            elif "features.7" in name or "features.6" in name:
                params_to_freeze.append(param)
            # Note 1: We did not observed any training performance improvement after unfreezing more layers.
            # Note 2: In case of CUDA availability uncomment the below lines for training.
            # elif 'features.5' in name or 'features.4' in name:
            #     params_backbone.append(param)
            # else:
            #     param.requires_grad = False
            else:
                params_backbone.append(param)
    else:
        print("Network is not ResNet or ConvNext.", flush=True)

    # Note: The "rbf_head" have the same implementation as the "pipnet_head" but instead of final ReLU layer a Softmax layer is used.
    if args.head_type in ("pipnet_head", "rbf_head"):
        classification_weight = []
        classification_bias = []
        for name, param in net.module._classification.named_parameters():
            if "weight" in name:
                classification_weight.append(param)
            elif "multiplier" in name:
                param.requires_grad = False
            else:
                if args.bias:
                    classification_bias.append(param)

    paramlist_net = [
        {
            "params": params_backbone,
            "lr": args.lr_net,
            "weight_decay_rate": args.weight_decay,
        },
        {
            "params": params_to_freeze,
            "lr": args.lr_block,
            "weight_decay_rate": args.weight_decay,
        },
        {
            "params": params_to_train,
            "lr": args.lr_block,
            "weight_decay_rate": args.weight_decay,
        },
        {
            "params": net.module._add_on.parameters(),
            "lr": args.lr_block,
            "weight_decay_rate": args.weight_decay,
        },
    ]

    if optimizer_type == "complete":
        paramlist_classifier = [
            {
                "params": params_backbone,
                "lr": args.lr_net,
                "weight_decay_rate": args.weight_decay,
            },
            {
                "params": params_to_freeze,
                "lr": args.lr_block,
                "weight_decay_rate": args.weight_decay,
            },
            {
                "params": params_to_train,
                "lr": args.lr_block,
                "weight_decay_rate": args.weight_decay,
            },
            {
                "params": net.module._add_on.parameters(),
                "lr": args.lr_block,
                "weight_decay_rate": args.weight_decay,
            },
            {
                "params": net.module._classification.parameters(),
                "lr": args.lr,
                "weight_decay_rate": args.weight_decay,
            },
        ]

    # paramlist_classifier when parameters from "cbc_head" are set to trainable.
    if args.head_type in ("cbc_head"):
        paramlist_classifier_only = [
            {
                "params": net.module._classification.parameters(),
                "lr": args.lr,
                "weight_decay_rate": args.weight_decay,
            },
        ]
    elif args.head_type in ("pipnet_head", "rbf_head"):
        paramlist_classifier_only = [
            {
                "params": classification_weight,
                "lr": args.lr,
                "weight_decay_rate": args.weight_decay,
            },
            {"params": classification_bias, "lr": args.lr, "weight_decay_rate": 0},
        ]

    if args.optimizer == "Adam":
        if optimizer_type == "separate":
            optimizer_net = torch.optim.AdamW(
                paramlist_net, lr=args.lr, weight_decay=args.weight_decay
            )
            optimizer_classifier_only = torch.optim.AdamW(
                paramlist_classifier_only, lr=args.lr, weight_decay=args.weight_decay
            )
            return (
                optimizer_net,
                optimizer_classifier_only,
                params_to_freeze,
                params_to_train,
                params_backbone,
            )
        elif optimizer_type == "complete":
            optimizer_net = torch.optim.AdamW(
                paramlist_net, lr=args.lr, weight_decay=args.weight_decay
            )
            optimizer_classifier = torch.optim.AdamW(
                paramlist_classifier, lr=args.lr, weight_decay=args.weight_decay
            )
            optimizer_classifier_only = torch.optim.AdamW(
                paramlist_classifier_only, lr=args.lr, weight_decay=args.weight_decay
            )
            return (
                optimizer_net,
                optimizer_classifier,
                optimizer_classifier_only,
                params_to_freeze,
                params_to_train,
                params_backbone,
            )
    else:
        raise ValueError("this optimizer type is not implemented")
