import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import yaml
from omegaconf import OmegaConf

logger = logging.getLogger()
logger.setLevel(logging.INFO)


@dataclass
class ModelConfig:
    dataset: str
    validation_size: float
    net: str
    batch_size: int
    batch_size_pretrain: int
    optimizer: str
    lr: float
    lr_block: float
    lr_net: float
    weight_decay: float
    disable_cuda: bool
    log_dir: str
    num_features: int
    image_size: int
    state_dict_dir_net: str
    dir_for_saving_images: str
    disable_pretrained: bool
    epochs_pretrain: int
    weighted_loss: bool
    seed: int
    gpu_ids: str
    num_workers: int
    bias: bool
    head_type: str
    notebook: Optional[bool] = None
    epochs_fine_tune: Optional[int] = None
    epochs_net: Optional[int] = None
    num_classes: Optional[int] = None
    lr_classifier_fine_tune: Optional[float] = None
    detect_prob_flag: Optional[bool] = None
    margin: Optional[float] = None
    align_pf_weight: Optional[float] = None
    softmax_pool: Optional[bool] = None
    reasoning_type: Optional[str] = None
    extra_test_image_folder: Optional[str] = None
    epochs: Optional[int] = None
    freeze_epochs: Optional[int] = None
    visualize_data: Optional[bool] = None
    wshape: Optional[int] = None
    eps: Optional[float] = None


def load_model_config(config_path: Path) -> OmegaConf:
    try:
        with open(config_path, "r") as f:
            config_yaml = yaml.safe_load(f)
            model_config_instance = ModelConfig(**config_yaml)
            return OmegaConf.structured(model_config_instance)
    except Exception as exception_message:
        logging.error(
            "Error in loading the model config file and exception error is stated below."
        )
        logging.error(exception_message)


if __name__ == "__main__":
    # demo test for loading the structured configuration instance
    logging.info(load_model_config(Path("cub_configs/cub_convnext_config_base.yaml")))
    logging.info(
        load_model_config(Path("cub_configs/cub_convnext_aligned_config_base.yaml"))
    )
