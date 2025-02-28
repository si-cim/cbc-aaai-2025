from pathlib import Path

# Defining the config path for the experiment runs.
CONF_BASE_DIR = Path("./deep_cbc/configs")

CONF_CARS_DIR = CONF_BASE_DIR / Path("car_configs")
CONF_CUB_DIR = CONF_BASE_DIR / Path("cub_configs")
CONF_PETS_DIR = CONF_BASE_DIR / Path("pets_configs")

CARS_CONVNEXT_ALIGN_CONF = CONF_CARS_DIR / Path(
    "cars_convnext_aligned_config_base.yaml"
)
CARS_CONVNEXT_CONF = CONF_CARS_DIR / Path("cars_convnext_config_base.yaml")
CARS_RBFNET_CONF = CONF_CARS_DIR / Path("cars_rbfnet_config_base.yaml")
CARS_RESNET_CONF = CONF_CARS_DIR / Path("cars_resnet_config_base.yaml")
CARS_PIPNET_CONF = CONF_CARS_DIR / Path("cars_pipnet_convnext_config_base.yaml")

CUB_CONVNEXT_ALIGN_CONF = CONF_CARS_DIR / Path("cub_convnext_aligned_config_base.yaml")
CUB_CONVNEXT_CONF = CONF_CARS_DIR / Path("cub_convnext_config_base.yaml")
CUB_RBFNET_CONF = CONF_CARS_DIR / Path("cub_rbfnet_config_base.yaml")
CUB_RESNET_CONF = CONF_CARS_DIR / Path("cub_resnet_config_base.yaml")
CUB_PIPNET_CONF = CONF_CARS_DIR / Path("cub_pipnet_convnext_config_base.yaml")

PETS_CONVNEXT_ALIGN_CONF = CONF_CARS_DIR / Path(
    "pets_convnext_aligned_config_base.yaml"
)
PETS_CONVNEXT_CONF = CONF_CARS_DIR / Path("pets_convnext_config_base.yaml")
PETS_RBFNET_CONF = CONF_CARS_DIR / Path("pets_rbfnet_config_base.yaml")
PETS_RESNET_CONF = CONF_CARS_DIR / Path("pets_resnet_config_base.yaml")
PETS_PIPNET_CONF = CONF_CARS_DIR / Path("pets_pipnet_convnext_config_base.yaml")


# Defining base path for the Tensorboard logging.
TENSORBOARD_LOG_DIR = Path("cbcnet_training_logs/")

# Defining paths of the interpretability benchmark datasets for the data loaders.
CUB_200_2011_PATH = Path("data/CUB_200_2011")
CUB_200_2011_TRAIN = Path("data/CUB_200_2011/dataset/train")
CUB_200_2011_TRAIN_CROPPED = Path("data/CUB_200_2011/dataset/train_crop")
CUB_200_2011_TEST = Path("data/CUB_200_2011/dataset/test_full")
CUB_200_2011_TEST_CROPPED = Path("data/CUB_200_2011/dataset/test_crop")
CUB_200_2011_IMAGES_PATH = Path("data/CUB_200_2011/dataset/images.txt")
CUB_200_2011_SPLIT_PATH = Path("data/CUB_200_2011/dataset/train_test_split.txt")
CUB_200_2011_BBOX_PATH = Path("data/CUB_200_2011/dataset/bounding_boxes.txt")

PETS_PATH = Path("data/PETS")
PETS_IMAGES_PATH = Path("data/PETS/images")
PETS_TRAIN = Path("data/PETS/dataset/train")
PETS_TEST = Path("data/PETS/dataset/test")
PETS_SPLIT_TRAIN_PATH = Path("data/PETS/annotations/trainval.txt")
PETS_SPLIT_TEST_PATH = Path("data/PETS/annotations/test.txt")

CARS_TRAIN = Path("data/cars/dataset/train")
CARS_TEST = Path("data/cars/dataset/test")
