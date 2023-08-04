#
# Copyright (C) 2023 Apple Inc. All rights reserved.
#
"""Methods to initialize and create dataset loaders."""
import os
import logging
from typing import Any, Dict, Tuple

from torch.utils.data import Dataset
import torchvision.datasets as datasets
import transforms


def download_dataset(data_path: str, dataset: str) -> None:
    """Download dataset prior to spawning workers.

    Args:
        data_path: Path to the root of the dataset.
        dataset: The name of the dataset.
    """
    if dataset == "imagenet":
        # ImageNet data requires manual download
        traindir = os.path.join(data_path, "training")
        valdir = os.path.join(data_path, "validation")
        assert os.path.isdir(
            traindir
        ), "Please download ImageNet training set to {}.".format(traindir)
        assert os.path.isdir(
            valdir
        ), "Please download ImageNet validation set to {}.".format(valdir)
    elif dataset == "cifar100":
        datasets.CIFAR100(root=data_path, train=True, download=True, transform=None)
    elif dataset == "flowers102":
        datasets.Flowers102(
            root=data_path, split="train", download=True, transform=None
        )
    elif dataset == "food101":
        datasets.Food101(root=data_path, split="train", download=True, transform=None)


def get_dataset_size(dataset: str) -> int:
    """Return dataset size to compute the number of iterations per epoch."""
    if dataset == "imagenet":
        return 1281167
    elif dataset == "cifar100":
        return 50000
    elif dataset == "flowers102":
        return 1020
    elif dataset == "food101":
        return 75750


def get_dataset_num_classes(dataset: str) -> int:
    """Return number of classes in a dataset."""
    if dataset == "imagenet":
        return 1000
    elif dataset == "cifar100":
        return 100
    elif dataset == "flowers102":
        return 102
    elif dataset == "food101":
        return 101


def get_dataset(config: Dict[str, Any]) -> Tuple[Dataset, Dataset]:
    """Return data loaders for training and validation sets."""
    logging.info("Instantiating {} dataset.".format(config["dataset"]))

    dataset_name = config.get("dataset", None)
    if dataset_name is None:
        logging.error("Dataset name can't be None")
    dataset_name = dataset_name.lower()

    if dataset_name == "imagenet":
        return get_imagenet_dataset(config)
    elif dataset_name == "cifar100":
        return get_cifar100_dataset(config)
    elif dataset_name == "flowers102":
        return get_flowers102_dataset(config)
    elif dataset_name == "food101":
        return get_food101_dataset(config)
    else:
        raise NotImplementedError


def get_cifar100_dataset(config) -> Tuple[Dataset, Dataset]:
    """Return training/test datasets for CIFAR-100 dataset.

    @TECHREPORT{Krizhevsky09learningmultiple,
        author = {Alex Krizhevsky},
        title = {Learning multiple layers of features from tiny images},
        institution = {},
        year = {2009}
    }
    """
    relative_path = config["data_path"]
    download_dataset = config.get("download_dataset", False)

    train_dataset = datasets.CIFAR100(
        root=relative_path,
        train=True,
        download=download_dataset,
        transform=transforms.compose_from_config(config["image_augmentation"]["train"]),
    )

    val_dataset = datasets.CIFAR100(
        root=relative_path,
        train=False,
        download=False,
        transform=transforms.compose_from_config(config["image_augmentation"]["val"]),
    )
    return train_dataset, val_dataset


def get_imagenet_dataset(config) -> Tuple[Dataset, Dataset]:
    """Return training/validation datasets for ImageNet dataset."""
    traindir = os.path.join(config["data_path"], "training")
    valdir = os.path.join(config["data_path"], "validation")

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.compose_from_config(config["image_augmentation"]["train"]),
    )

    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.compose_from_config(config["image_augmentation"]["val"]),
    )
    return train_dataset, val_dataset


def get_flowers102_dataset(config) -> Tuple[Dataset, Dataset]:
    """Return training/test datasets for Flowers-102 dataset.

    @InProceedings{Nilsback08,
       author = "Nilsback, M-E. and Zisserman, A.",
       title = "Automated Flower Classification over a Large Number of Classes",
       booktitle = "Proceedings of the Indian Conference on Computer Vision, Graphics
        and Image Processing",
       year = "2008",
       month = "Dec"
    }
    """
    relative_path = config["data_path"]
    download_dataset = config.get("download_dataset", False)

    train_dataset = datasets.Flowers102(
        root=relative_path,
        split="train",
        download=download_dataset,
        transform=transforms.compose_from_config(config["image_augmentation"]["train"]),
    )
    val_dataset = datasets.Flowers102(
        root=relative_path,
        split="test",
        download=False,
        transform=transforms.compose_from_config(config["image_augmentation"]["val"]),
    )
    return train_dataset, val_dataset


def get_food101_dataset(config) -> Tuple[Dataset, Dataset]:
    """Return training/test datasets for Food-101 dataset.

    @inproceedings{bossard14,
      title = {Food-101 -- Mining Discriminative Components with Random Forests},
      author = {Bossard, Lukas and Guillaumin, Matthieu and Van Gool, Luc},
      booktitle = {European Conference on Computer Vision},
      year = {2014}
    }
    """
    relative_path = config["data_path"]
    download_dataset = config.get("download_dataset", False)

    train_dataset = datasets.Food101(
        root=relative_path,
        split="train",
        download=download_dataset,
        transform=transforms.compose_from_config(config["image_augmentation"]["train"]),
    )
    val_dataset = datasets.Food101(
        root=relative_path,
        split="test",
        download=False,
        transform=transforms.compose_from_config(config["image_augmentation"]["val"]),
    )
    return train_dataset, val_dataset
