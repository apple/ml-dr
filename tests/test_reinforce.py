#
# Copyright (C) 2023 Apple Inc. All rights reserved.
#

"""Tests for dataset generation."""

from typing import Any, Dict
import pytest
import torch

import dr.transforms as T_dr
from reinforce import transform_batch

# from torchvision.transforms import InterpolationMode, functional as F


def single_transform_config_test(
    config: Dict[str, Any],
    num_images: int,
    num_samples: int,
    height: int,
    width: int,
    crop_height: int,
    crop_width: int,
    compress: bool,
    single_image_test: bool,
) -> None:
    """Test applying and reapplying transformations to a batch of images."""
    ids = torch.arange(num_images)
    images_orig = torch.rand(size=(num_images, 3, height, width))
    images = torch.zeros(num_images, num_samples, 3, crop_height, crop_width)
    target = torch.randint(low=0, high=10, size=(num_images,))
    # Simulate data loader
    transforms = T_dr.compose_from_config(
        T_dr.before_collate_config(config["reinforce"]["image_augmentation"])
    )

    images = []
    coords = []
    for i in range(num_images):
        sample_all, params_all = T_dr.before_collate_apply(
            images_orig[i], transforms, num_samples
        )
        images += [sample_all]
        coords += [params_all]
    images = torch.utils.data.default_collate(images)
    coords = torch.utils.data.default_collate(coords)

    # Apply after collate transformations
    transforms = T_dr.compose_from_config(config["reinforce"]["image_augmentation"])
    images_aug, params_aug = transform_batch(
        ids, images, target, coords, transforms, config
    )
    assert images_aug.shape == (
        num_images,
        num_samples,
        3,
        crop_height,
        crop_width,
    ), "Incorrect shape of transformed image."
    transforms = T_dr.compose_from_config(config["reinforce"]["image_augmentation"])

    # Single test
    # TODO: support transforms(img) and reapply(img, params) without img2
    if single_image_test:
        img, param = transforms(images_orig[0], images_orig[0])
        img2, param2 = transforms.reapply(images_orig[0], param)
        torch.testing.assert_close(
            actual=img,
            expected=img2,
        )

    if num_images > 1 and num_samples > 1:
        assert (
            len(set([str(q) for p in params_aug for q in p])) > 1
        ), "Parameters are not random."

    # Test reapply
    for i in range(num_images):
        for j in range(num_samples):
            if compress:
                params_aug[i][j] = transforms.decompress(params_aug[i][j])
            img2 = None
            if "mixup" in params_aug[i][j] or "cutmix" in params_aug[i][j]:
                img2 = images_orig[params_aug[i][j]["id2"]]
            out, _ = transforms.reapply(images_orig[i], params_aug[i][j], img2)
            torch.testing.assert_close(actual=out, expected=images_aug[i][j])


@pytest.mark.parametrize("compress", [False, True])
def test_random_resized_crop(compress: bool) -> None:
    """Test RRC with other transformations."""
    num_images = 2
    num_samples = 4
    height = 10
    width = 10
    crop_height = 5
    crop_width = 5
    single_image_test = True
    config = {
        "reinforce": {
            "num_samples": num_samples,
            "compress": compress,
            "image_augmentation": {
                "uint8": {"enable": True},
                "random_resized_crop": {
                    "enable": True,
                    "size": [crop_height, crop_width],
                },
                "random_horizontal_flip": {"enable": True, "p": 0.5},
                "rand_augment": {"enable": True, "p": 0.5},
                "to_tensor": {"enable": True},
                "normalize": {
                    "enable": True,
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225],
                },
                "random_erase": {"enable": True, "p": 0.25},
            },
        }
    }
    single_transform_config_test(
        config,
        num_images,
        num_samples,
        height,
        width,
        crop_height,
        crop_width,
        compress,
        single_image_test,
    )


@pytest.mark.parametrize("compress", [False, True])
def test_center_crop(compress: bool) -> None:
    """Test center-crop with other transformations."""
    num_images = 2
    num_samples = 4
    height = 10
    width = 10
    crop_height = 5
    crop_width = 5
    single_image_test = True
    config = {
        "reinforce": {
            "num_samples": num_samples,
            "compress": compress,
            "image_augmentation": {
                "uint8": {"enable": True},
                "center_crop": {"enable": True, "size": [crop_height, crop_width]},
                "random_horizontal_flip": {"enable": True, "p": 0.5},
                "rand_augment": {"enable": True, "p": 0.5},
                "to_tensor": {"enable": True},
                "normalize": {
                    "enable": True,
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225],
                },
                "random_erase": {"enable": True, "p": 0.25},
            },
        }
    }
    single_transform_config_test(
        config,
        num_images,
        num_samples,
        height,
        width,
        crop_height,
        crop_width,
        compress,
        single_image_test,
    )


@pytest.mark.parametrize("compress", [False, True])
def test_mixing(compress: bool) -> None:
    """Test MixUp/CutMix with other transformations."""
    num_images = 10
    num_samples = 20
    height = 256
    width = 256
    crop_height = 224
    crop_width = 224
    single_image_test = False
    config = {
        "reinforce": {
            "num_samples": num_samples,
            "compress": compress,
            "image_augmentation": {
                "uint8": {"enable": True},
                "random_resized_crop": {
                    "enable": True,
                    "size": [crop_height, crop_width],
                },
                "random_horizontal_flip": {"enable": True, "p": 0.5},
                "rand_augment": {"enable": True, "p": 0.5},
                "to_tensor": {"enable": True},
                "normalize": {
                    "enable": True,
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225],
                },
                "random_erase": {"enable": True, "p": 0.25},
                "mixup": {"enable": True, "alpha": 1.0, "p": 0.5},
                "cutmix": {"enable": True, "alpha": 1.0, "p": 0.5},
            },
        }
    }
    single_transform_config_test(
        config,
        num_images,
        num_samples,
        height,
        width,
        crop_height,
        crop_width,
        compress,
        single_image_test,
    )
