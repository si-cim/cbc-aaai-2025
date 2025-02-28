import logging
import os
import shutil
import time
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from deep_cbc.utils.constants import *


def preprocess_cub() -> None:
    """Preprocessing script for creating equally cropped CUB dataset images for training."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    path = Path("./" + CUB_200_2011_PATH)

    time_start = time.time()

    path_images = Path("./" + CUB_200_2011_IMAGES_PATH)
    path_split = Path("./" + CUB_200_2011_SPLIT_PATH)
    train_save_path = Path("./" + CUB_200_2011_TRAIN_CROPPED)
    test_save_path = Path("./" + CUB_200_2011_TEST_CROPPED)
    bbox_path = Path("./" + CUB_200_2011_BBOX_PATH)

    images = []
    with open(path_images, "r") as f:
        for line in f:
            images.append(list(line.strip("\n").split(",")))
    logging.info("Images: ", images)
    split = []
    with open(path_split, "r") as f_:
        for line in f_:
            split.append(list(line.strip("\n").split(",")))

    bboxes = dict()
    with open(bbox_path, "r") as bf:
        for line in bf:
            id, x, y, w, h = tuple(map(float, line.split(" ")))
            bboxes[int(id)] = (x, y, w, h)

    num = len(images)
    for k in range(num):
        id, fn = images[k][0].split(" ")
        id = int(id)
        file_name = fn.split("/")[0]
        if int(split[k][0][-1]) == 1:

            if (train_save_path / Path(file_name)).is_dir():
                (train_save_path / Path(file_name)).mkdir(parents=True, exist_ok=True)

            img = Image.open(
                path / Path("images") / Path(images[k][0].split(" ")[1])
            ).convert("RGB")
            x, y, w, h = bboxes[id]
            cropped_img = img.crop((x, y, x + w, y + h))
            cropped_img.save(
                train_save_path
                / Path(file_name)
                / Path(images[k][0].split(" ")[1].split("/")[1])
            )
            logging.info("%s" % images[k][0].split(" ")[1].split("/")[1])
        else:
            if not (test_save_path / Path(file_name)).is_dir():
                (test_save_path / Path(file_name)).mkdir(parents=True, exist_ok=True)

            img = Image.open(
                path / Path("images") / Path(images[k][0].split(" ")[1])
            ).convert("RGB")
            x, y, w, h = bboxes[id]
            cropped_img = img.crop((x, y, x + w, y + h))
            cropped_img.save(
                test_save_path
                / Path(file_name)
                / Path(images[k][0].split(" ")[1].split("/")[1])
            )
            logging.info("%s" % images[k][0].split(" ")[1].split("/")[1])

    train_save_path = Path("./" + CUB_200_2011_TRAIN)
    test_save_path = Path("./" + CUB_200_2011_TEST)

    num = len(images)
    for k in range(num):
        id, fn = images[k][0].split(" ")
        id = int(id)
        file_name = fn.split("/")[0]
        if int(split[k][0][-1]) == 1:

            if not (train_save_path / Path(file_name)).is_dir():
                (train_save_path / Path(file_name)).mkdir(parents=True, exist_ok=True)
            img = Image.open(
                path / Path("images") / Path(images[k][0].split(" ")[1])
            ).convert("RGB")
            width, height = img.size

            img.save(
                train_save_path
                / Path(file_name)
                / Path(images[k][0].split(" ")[1].split("/")[1])
            )

            logging.info("%s" % images[k][0].split(" ")[1].split("/")[1])
        else:
            if not (test_save_path / Path(file_name)).is_dir():
                (test_save_path / Path(file_name)).mkdir(parents=True, exist_ok=True)
            shutil.copy(
                path / Path("images") / Path(images[k][0].split(" ")[1]),
                test_save_path
                / Path(file_name)
                / Path(images[k][0].split(" ")[1].split("/")[1]),
            )
            logging.info("%s" % images[k][0].split(" ")[1].split("/")[1])
    time_end = time.time()
    logging.info("CUB200, %s!" % (time_end - time_start))


def preprocess_pets() -> None:
    """Preprocessing script for creating equally cropped PETS dataset images for training."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    pet_data_path = Path("./" + PETS_PATH)
    train_list_path = Path("./" + PETS_SPLIT_TRAIN_PATH)
    test_list_path = Path("./" + PETS_SPLIT_TEST_PATH)
    images_path = Path("./" + PETS_IMAGES_PATH)
    train_path = Path("./" + PETS_TRAIN)
    test_path = Path("./" + PETS_TEST)

    train_images = []  # list of train images
    with open(train_list_path, "r") as f:
        for line in f:
            line_data = line.strip("\n").split(" ")
            train_images.append((line_data[0], int(line_data[1])))

    test_images = []  # list of test images
    with open(test_list_path, "r") as f:
        for line in f:
            line_data = line.strip("\n").split(" ")
            test_images.append((line_data[0], int(line_data[1])))

    train_path.mkdir(parents=True, exist_ok=True)  # make train directory
    test_path.mkdir(parents=True, exist_ok=True)  # make test directory

    train_time_start = time.time()
    for image, target in tqdm(train_images):  # preparing the train dataset
        target_dir_path = train_path / image.rsplit("_", 1)[0]
        target_dir_path.mkdir(parents=True, exist_ok=True)
        shutil.copy(
            images_path / Path(image + ".jpg"), target_dir_path / Path(image + ".jpg")
        )
    train_time_end = time.time()
    logging.info(
        "PETS Train Split Preprocessing Time: %s!" % (train_time_end - train_time_start)
    )

    test_time_start = time.time()
    for image, target in tqdm(test_images):  # preparing the test dataset
        target_dir_path = test_path / image.rsplit("_", 1)[0]
        target_dir_path.mkdir(parents=True, exist_ok=True)
        shutil.copy(
            images_path / Path(image + ".jpg"), target_dir_path / Path(image + ".jpg")
        )
    test_time_end = time.time()
    logging.info(
        "PETS Test Split Preprocessing Time: %s!" % (test_time_end - test_time_start)
    )
