#
# Copyright (C) 2023 Apple Inc. All rights reserved.
#
"""Methods to initialize and create dataset loaders."""
import os
import random
from typing import Union, Tuple, Any, List, Dict

import torch
from torch import Tensor
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision
from torch.utils.data import Dataset
import torch.utils.data.distributed
import dr.transforms as T_dr
from dr.utils import densify
from data import get_dataset_num_classes


class ReinforceMetadata:
    """A class to load and return only the metadata of the reinforced dataset."""

    def __init__(self, rdata_path: Union[str, List[str]]) -> None:
        """Iniatilize the metadata files and configuration."""
        self.rdata_path = rdata_path if isinstance(rdata_path, list) else [rdata_path]
        assert any(
            [os.path.exists(rp) for rp in rdata_path]
        ), f"Please download reinforce metadata to {rdata_path}."
        self.rconfig = self.get_rconfig()

    def __getitem__(self, index: int) -> Tuple[int, List[List[float]]]:
        """Return reinforced metadata for a single data point at given index."""
        p = random.randint(0, len(self.rdata_path) - 1)
        rdata = torch.load(os.path.join(self.rdata_path[p], "{}.pth.tar".format(index)))
        return rdata

    def get_rconfig(self) -> Dict[str, Any]:
        """Read configuration file for reinforcements."""
        num_samples = 0
        for rdata in self.rdata_path:
            rconfig = torch.load(os.path.join(rdata, "config.pth.tar"))
            num_samples += rconfig["reinforce"]["num_samples"]
        rconfig["reinforce"]["num_samples"] = num_samples
        return rconfig


class ReinforcedDataset(Dataset):
    """A class to reinforce a given dataset with metadata in rdata_path."""

    def __init__(
        self,
        dataset: Dataset,
        rdata_path: Union[str, List[str]],
        config: Dict[str, Any],
        num_classes: int,
    ) -> None:
        """Initialize the metadata configuration and parameters."""
        self.ds = dataset
        self.num_classes = num_classes
        self.densify_method = config.get("densify", "zeros")
        self.p = config.get("p", 1.0) or 1.0
        self.config = config

        # Use specified data augmentations only if sampling the original data
        self.transform_orig = dataset.transform
        dataset.transform = None

        self.r_metadata = ReinforceMetadata(rdata_path)
        rconfig = self.r_metadata.get_rconfig()

        # Initialize transformations from config
        self.transforms = T_dr.compose_from_config(
            rconfig["reinforce"]["image_augmentation"]
        )

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the sample from the dataset with given index."""
        # With probability self.p sample reinforced data, otherwise sample original
        p = random.random()

        if p < self.p:
            # Load reinforcement meta data for the image
            rdata = self.r_metadata[index]

            # Choose a random reinforcement
            assert rdata[0] == index, "Index does not match the metadata index."
            rdata = rdata[1]  # tuple (id, rdata)
            i = random.randint(0, len(rdata) - 1)
            rdata_sample = rdata[i]
            params, target = rdata_sample["params"], rdata_sample["prob"]
            if isinstance(params, list):
                params = self.transforms.decompress(params)

            # Load image
            img, _ = self.ds[index]

            # Load the image pair if reinforcement has mixup/cutmix
            img2 = None
            if "cutmix" in params or "mixup" in params:
                img2, _ = self.ds[params["id2"]]

            # Reapply augmentation
            img, reparams = self.transforms.reapply(img, params, img2)
            for k, v in reparams.items():
                assert v == params[k], "params changed."
            target = densify(target, self.num_classes, self.densify_method)
        else:
            # With probability 1-self.p sample the original data
            img, target = self.ds[index]
            if self.transform_orig:
                img = self.transform_orig(img)
            target = torch.nn.functional.one_hot(
                torch.tensor(target), num_classes=self.num_classes
            ).float()
        return img.detach(), target.detach()

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.ds)


class DatasetWithParameters:
    """A wrapper to the PyTorch datasets that transformation parameters."""

    def __init__(
        self,
        dataset: torchvision.datasets.VisionDataset,
        transform: T_dr.Compose,
        num_samples: int,
    ) -> None:
        """Initialize and set the number of random crops per sample."""
        self.ds = dataset
        self.num_samples = num_samples
        self.transform = transform
        self.ds.transform = None

    def __getitem__(self, index: int) -> Tuple[Tensor, int, Tensor]:
        """Return multiple random transformations of a sample at given index.

        Args:
            index: An integer that is the unique ID of a sample in the dataset.

        Returns:
            A Tuple of (inputs, target, params) of shape:
                sample_all: [num_samples,]+sample.shape
                target: int
                params: A dictionary of parameters, each value is a Tensor of shape
                    [num,_samples, ...]
        """
        sample, target = self.ds[index]
        sample_all, params_all = T_dr.before_collate_apply(
            sample, self.transform, self.num_samples
        )
        return sample_all, target, params_all

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.ds)


class IndexedDataset(torch.utils.data.Dataset):
    """A wrapper to PyTorch datasets that returns an index."""

    def __init__(self, dataset: torch.utils.data.Dataset) -> None:
        """Set the dataset."""
        self.ds = dataset

    def __getitem__(self, index: int) -> Tuple[int, ...]:
        """Return a sample and the given index."""
        data = self.ds[index]
        return (index, *data)

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.ds)
