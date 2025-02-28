import argparse
import logging
import pickle
from pathlib import Path
from typing import Union

from omegaconf import DictConfig, OmegaConf

from deep_cbc.configs.model_config import ModelConfig

"""
    Utility functions for handling parsed arguments
"""

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Train a PIPNet, RBFNet or CBCNet Prototype Based Network."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="CUB-200-2011",
        help="Data set on which the model should be trained.",
    )
    parser.add_argument(
        "--validation_size",
        type=float,
        default=0.0,
        help="Split between training and validation set. Can be zero when there is a separate test or validation directory. Should be between 0 and 1. Used for partimagenet (e.g. 0.2).",
    )
    parser.add_argument(
        "--net",
        type=str,
        default="convnext_tiny_26",
        help="Base network used as backbone of the Prototype Based Network Model. Default is convnext_tiny_26 with adapted strides to output 26x26 latent representations. Other option is convnext_tiny_13 that outputs 13x13 (smaller and faster to train, less fine-grained). Pretrained network on iNaturalist is only available for resnet50_inat. Options are: resnet18, resnet34, resnet50, resnet50_inat, resnet101, resnet152, convnext_tiny_26 and convnext_tiny_13.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size when training the model using minibatch gradient descent. Batch size is multiplied with number of available GPUs.",
    )
    parser.add_argument(
        "--batch_size_pretrain",
        type=int,
        default=128,
        help="Batch size when pretraining the prototypes (first training stage).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=60,
        help="The number of epochs the Prototype Based Network should be trained (second training stage).",
    )
    parser.add_argument(
        "--epochs_pretrain",
        type=int,
        default=10,
        help="Number of epochs to pre-train the prototypes (first training stage). Recommended to train at least until the align loss < 1.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="Adam",
        help="The optimizer that should be used when training the Prototype Based Network.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.05,
        help="The optimizer learning rate for training the weights from prototypes to classes.",
    )
    parser.add_argument(
        "--lr_block",
        type=float,
        default=0.0005,
        help="The optimizer learning rate for training the last conv layers of the backbone.",
    )
    parser.add_argument(
        "--lr_net",
        type=float,
        default=0.0005,
        help="The optimizer learning rate for the backbone. Usually similar as lr_block.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="Weight decay used in the optimizer.",
    )
    parser.add_argument(
        "--disable_cuda",
        action="store_true",
        help="Flag that disables GPU usage if set.",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="./runs/run_pipnet",
        help="The directory in which train progress should be logged.",
    )
    parser.add_argument(
        "--num_features",
        type=int,
        default=0,
        help="Number of prototypes. When zero (default) the number of prototypes is the number of output channels of backbone. If this value is set, then a 1x1 conv layer will be added. Recommended to keep 0, but can be increased when number of classes > num output channels in backbone.",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=224,
        help="Input images will be resized to --image_size x --image_size (square). Code only tested with 224x224, so no guarantees that it works for different sizes.",
    )
    parser.add_argument(
        "--state_dict_dir_net",
        type=str,
        default="",
        help="The directory containing a state dict with a pretrained Prototype Based Network. E.g., ./runs/run_pipnet/checkpoints/net_pretrained.",
    )
    parser.add_argument(
        "--freeze_epochs",
        type=int,
        default=10,
        help="Number of epochs where pretrained features_net will be frozen while training classification layer (and last layer(s) of backbone).",
    )
    parser.add_argument(
        "--dir_for_saving_images",
        type=str,
        default="visualization_results",
        help="Directory for saving the prototypes and explanations.",
    )
    parser.add_argument(
        "--disable_pretrained",
        action="store_true",
        help="When set, the backbone network is initialized with random weights instead of being pretrained on another dataset).",
    )
    parser.add_argument(
        "--weighted_loss",
        action="store_true",
        help="Flag that weights the loss based on the class balance of the dataset. Recommended to use when data is imbalanced.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed. Note that there will still be differences between runs due to nondeterminism. Refer https://pytorch.org/docs/stable/notes/randomness.html.",
    )
    parser.add_argument(
        "--gpu_ids",  # Note: Currently, training over a single GPU is programmed only.
        type=str,
        default="0",  # TODO: Add support for training across multiple GPUs.
        help="ID of the GPU, can be separated with comma.",
    )
    parser.add_argument(
        "--num_workers", type=int, default=8, help="Num workers in dataloaders."
    )
    parser.add_argument(
        "--bias",
        action="store_true",
        help="Flag that indicates whether to include a trainable bias in the linear classification layer.",
    )
    parser.add_argument(
        "--extra_test_image_folder",
        type=str,
        default="./experiments",
        help="Folder with images that Prototype Based Networks will predict and explain, that are not in the training or test set. E.g. images with 2 objects or OOD image. Images should be in subfolder. E.g. images in ./experiments/images/, and argument --./experiments.",
    )
    parser.add_argument(
        "--notebook",  # Note: This flag is/was only relevant for the defined dataset paths
        type=bool,
        default=False,
        help="Whether the argument are being parsed through either a notebook or script.",
    )
    parser.add_argument(
        "--head_type",
        type=str,
        default="pipnet_head",
        choices=["cbc_head", "rbf_head", "pipnet_head"],
        help="The prototype head constraint for training the Prototype Based Network Model.",
    )
    parser.add_argument(
        "--softmax_pool",
        type=bool,
        default=False,
        help="This flag introduces a softmax constraint on the PIPNet model to get RBFNet which can also be considered equivalent to CBCNet with only positive reasoning.",
    )
    parser.add_argument(
        "--reasoning_type",
        type=str,
        default=None,
        help="This flag selects the reasoning vector while plotting the positive and negative prototype visualization.",
    )
    parser.add_argument(
        "--visualize_data",
        type=bool,
        default=False,
        help="This flag selects whether the visualization function will be executed while running the model trainer.",
    )

    args = parser.parse_args()
    if len(args.log_dir.split("/")) > 2:
        if not Path(args.log_dir).exists():
            Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    return args


def args_to_config_struct(args: argparse.Namespace) -> OmegaConf:
    args_dict = vars(args)
    model_config_instance = ModelConfig(**args_dict)
    return OmegaConf.structured(model_config_instance)


def save_args(args: argparse.Namespace, directory_path: Union[str, Path]) -> None:
    """
    Save the arguments in the specified directory as
        - a text file called 'experiment_args.txt'
        - a pickle file called 'experiment_args.pickle'
    :param args: The arguments to be saved
    :param directory_path: The path to the directory where the arguments should be saved
    """
    # If the specified directory does not exists, create it.
    if not Path(directory_path).is_dir():
        Path(directory_path).mkdir(parents=True, exist_ok=True)

    # Save the args in a text file.
    with open(Path(directory_path) / Path("experiment_args.txt"), "w") as f:
        for arg in vars(args):
            val = getattr(args, arg)
            if isinstance(
                val, str
            ):  # Add quotation marks to indicate that the argument is of string type.
                val = f"'{val}'"
            f.write("{}: {}\n".format(arg, val))
    # Pickle the args for possible reuse.
    with open(Path(directory_path) / Path("experiment_args.pickle"), "wb") as f:
        pickle.dump(args, f)


def save_config(
    config: Union[OmegaConf, DictConfig], directory_path: Union[str, Path]
) -> None:
    """
    Save the configurations in the specified directory as
        - a text file called 'experiment_config.txt'
        - a pickle file called 'experiment_config.pickle'
    :param config: The experiment config to be saved in the directory
    :param directory_path: The path to the directory where the experiment config should be saved
    """
    OmegaConf.save(config, Path(directory_path) / "experiment_config.yaml")
    # Pickle the config for possible reuse
    with open(Path(directory_path) / "experiment_config.pickle", "wb") as f:
        pickle.dump(config, f)


if __name__ == "__main__":
    # demo test for checking whether the args namespace gets converted into OmegaConf struct
    args = get_args()
    args_struct = args_to_config_struct(args)
    logging.info(args_struct)
    save_config(args_struct, ".")
