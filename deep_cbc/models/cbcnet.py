import argparse
import logging
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from features.convnext_features import (
    convnext_tiny_13_features,
    convnext_tiny_26_features,
)
from features.resnet_features import (
    resnet18_features,
    resnet34_features,
    resnet50_features,
    resnet50_features_inat,
    resnet101_features,
    resnet152_features,
)
from omegaconf import OmegaConf
from torch import Tensor

from deep_cbc.models.cbc import SlimClassificationByComponents

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class CBCNet(nn.Module):
    def __init__(
        self,
        num_classes: int,
        num_prototypes: int,
        feature_net: nn.Module,
        args: Union[argparse.Namespace, OmegaConf],
        add_on_layers: nn.Module,
        pool_layer: nn.Module,
        classification_layer: nn.Module,
    ):
        super().__init__()
        assert num_classes > 0
        self._args = args
        self._num_features = args.num_features
        self._num_classes = num_classes
        self._num_prototypes = num_prototypes
        self._net = feature_net
        self._add_on = add_on_layers
        self._pool = pool_layer
        self._classification = classification_layer
        self._multiplier = classification_layer.normalization_multiplier

    def forward(self, xs, inference=False):
        features = self._net(xs)
        proto_features = self._add_on(features)
        pooled = self._pool(proto_features)
        if inference:
            # During inference, ignore all prototypes that have 0.1 similarity or lower.
            # Note: We believe this manual clipping of weight values across the PIPNet implementation
            # enforces sparsity manually. And the sparsity is not a byproduct of learning with PIPNet and
            # RBFNet implement over PIPNet by replacing the ReLU layer with a Softmax.
            clamped_pooled = torch.where(pooled < 0.1, 0.0, pooled)
            out, _ = self._classification(clamped_pooled)  # Shape: (bs*2, num_classes)
            return proto_features, clamped_pooled, out
        else:
            out, _ = self._classification(pooled)  # Shape: (bs*2, num_classes)
            return proto_features, pooled, out


base_architecture_to_features = {
    "resnet18": resnet18_features,  # The "args.net" select this backbone architecture.
    "resnet34": resnet34_features,
    "resnet50": resnet50_features,
    "resnet50_inat": resnet50_features_inat,
    "resnet101": resnet101_features,
    "resnet152": resnet152_features,
    "convnext_tiny_26": convnext_tiny_26_features,
    "convnext_tiny_13": convnext_tiny_13_features,
}


def clip(x: torch.Tensor) -> torch.Tensor:
    return torch.clip(x, min=0, max=1)


def init_cbc_head(num_prototypes: int, num_classes: int, device: torch.device = None):
    """
    This function initializes the CBC classification head with reasoning labels
    and probabilities to learn positive and negative reasoning.
    """
    reasoning_labels = (
        torch.nn.functional.one_hot(
            torch.tensor(list(range(num_classes)), dtype=torch.int64),
            num_classes=num_classes,
        )
        .float()
        .to(device)
    )

    init_reasoning_probabilities = (
        0.6 - torch.randn(2 * num_prototypes, num_classes) * 0.2
    ).to(device)

    classification_head = SlimClassificationByComponents(
        reasoning_labels=reasoning_labels,
        init_reasoning_probabilities=init_reasoning_probabilities,
    ).to(device)

    return classification_head


# adapted from https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear
class NonNegLinear(nn.Module):
    """Applies a linear transformation to the incoming data with non-negative weights."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        softmax_pool: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(NonNegLinear, self).__init__()
        self.softmax_pool = softmax_pool
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        self.normalization_multiplier = nn.Parameter(
            torch.ones((1,), requires_grad=True)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

    def forward(self, input: Tensor) -> Tensor:
        if self.softmax_pool:
            return F.linear(input, torch.softmax(self.weight, dim=1), self.bias)

        return F.linear(input, torch.relu(self.weight), self.bias)


def get_network(num_classes: int, args: Union[argparse.Namespace, OmegaConf]):
    features = base_architecture_to_features[args.net](
        pretrained=not args.disable_pretrained
    )
    features_name = str(features).upper()
    if "next" in args.net:
        features_name = str(args.net).upper()
    if features_name.startswith("RES") or features_name.startswith("CONVNEXT"):
        first_add_on_layer_in_channels = [
            i for i in features.modules() if isinstance(i, nn.Conv2d)
        ][-1].out_channels
    else:
        raise Exception("other base architecture NOT implemented")

    if args.num_features == 0:
        num_prototypes = first_add_on_layer_in_channels
        logging.info("Number of prototypes: " + str(num_prototypes))
        add_on_layers = nn.Sequential(
            nn.Softmax(dim=1),
            # Softmax over every prototype for each patch, such that for every location in image,
            # summation over prototypes is 1. Hence, across depth "D" the Softmax is applied.
        )
    else:
        num_prototypes = args.num_features
        logging.info(
            "Number of prototypes set from "
            + str(first_add_on_layer_in_channels)
            + " to "
            + str(num_prototypes)
            + ". Extra 1x1 conv layer added. Not recommended."
        )
        add_on_layers = nn.Sequential(
            nn.Conv2d(
                in_channels=first_add_on_layer_in_channels,
                out_channels=num_prototypes,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
            nn.Softmax(dim=1),  # Shape: (bs, ps or D,H,W)
            # Softmax over every prototype for each patch, such that for every location in image,
            # summation over prototypes is 1. Hence, across depth "D" the Softmax is applied.
        )

    pool_layer = nn.Sequential(
        nn.AdaptiveMaxPool2d(output_size=(1, 1)),  # Outputs: (bs, ps,1,1)
        nn.Flatten(),  # Outputs: (bs, ps)
    )

    if args.head_type == "rbf_head":
        classification_layer = NonNegLinear(
            num_prototypes, num_classes, bias=args.bias, softmax_pool=True
        )
    elif args.head_type == "pipnet_head":
        classification_layer = NonNegLinear(
            num_prototypes, num_classes, bias=args.bias, softmax_pool=False
        )
    elif args.head_type == "cbc_head":
        classification_layer = init_cbc_head(
            num_prototypes,
            num_classes,
            torch.device("cuda:" + str(args.gpu_ids).split(",")[0]),
        )

    return features, add_on_layers, pool_layer, classification_layer, num_prototypes
