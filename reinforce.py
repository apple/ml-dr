#
# Copyright (C) 2023 Apple Inc. All rights reserved.
#
"""Reinforce a dataset given a pretrained teacher and save the metadata."""

import os
import time
import argparse
from typing import Any, Dict, Tuple
import yaml
import random
import warnings
import copy
import joblib

import torch
import torch.nn.parallel
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import dr.transforms as T_dr
from dr.data import IndexedDataset, DatasetWithParameters
from dr.utils import sparsify
import internal
import internal.hooks
import internal.data
from models import load_model

from utils import ProgressMeter, AverageMeter
from data import (
    download_dataset,
    get_dataset,
    get_dataset_num_classes,
)

import logging

logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="PyTorch Reinforcing")
    parser.add_argument(
        "--config", type=str, required=False, help="Path to a yaml config file."
    )
    args = parser.parse_args()
    return args


def main(args) -> None:
    """Reinforce a model with the configurations specified in given arguments."""
    if args.config is not None:
        # Read parameters from yaml config for local run
        yaml_path = args.config
        with open(yaml_path, "r") as file:
            config = yaml.safe_load(file).get("parameters")

    dataset = config.get("dataset", "imagenet")
    config["num_classes"] = get_dataset_num_classes(dataset)

    if config["seed"] is not None:
        random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        cudnn.deterministic = True
        cudnn.benchmark = False
        warnings.warn(
            "You have chosen to seed training. "
            "This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! "
            "You may see unexpected behavior when restarting "
            "from checkpoints."
        )

    ngpus_per_node = torch.cuda.device_count()
    if config["gpu"] is not None:
        warnings.warn(
            "You have chosen a specific GPU. This will completely "
            "disable data parallelism."
        )
        ngpus_per_node = 1

    if config["dist_url"] == "env://" and config["world_size"] == -1:
        config["world_size"] = int(os.environ["WORLD_SIZE"])

    config["distributed"] = (
        config["world_size"] > 1 or config["multiprocessing_distributed"]
    )

    if config["download_data"]:
        config["data_path"] = download_dataset(config["data_path"], config["dataset"])
    if config["multiprocessing_distributed"]:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        config["world_size"] = ngpus_per_node * config["world_size"]
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(
            main_worker,
            nprocs=ngpus_per_node,
            args=(ngpus_per_node, copy.deepcopy(config)),
        )

    else:
        # Simply call main_worker function
        main_worker(config["gpu"], ngpus_per_node, copy.deepcopy(config))

    rdata_ext_path = os.path.join("/mnt/reinforced_data")
    fnames = next(os.walk(rdata_ext_path))[2]
    max_id = max(
        [
            int(f.replace(".pth.tar", "").replace(".jb", ""))
            for f in fnames
            if (".pth.tar" in f or ".jb" in f) and f != "config.pth.tar"
        ]
    )
    assert max_id + 1 == len(fnames), "Saved {} files but max id is {}.".format(
        len(fnames), max_id + 1
    )
    logging.info("Saving new data.")
    torch.save(config, os.path.join(rdata_ext_path, "config.pth.tar"))
    artifact_path = config["artifact_path"]
    gzip = config["reinforce"]["gzip"]
    cmd = "tar c{}f {} -C {} reinforced_data".format(
        "z" if gzip else "",
        os.path.join(
            artifact_path, "reinforced_data.tar{}".format(".gz" if gzip else "")
        ),
        "/mnt/",
    )
    os.system(cmd)


def main_worker(gpu: int, ngpus_per_node: int, config: Dict[str, Any]) -> None:
    """Reinforce data with a single process. In distributed training, run on one GPU."""
    config["gpu"] = gpu

    if config["gpu"] is not None:
        logging.info("Use GPU: {} for datagen".format(config["gpu"]))

    if config["distributed"]:
        if config["dist_url"] == "env://" and config["rank"] == -1:
            config["rank"] = int(os.environ["RANK"])
        if config["multiprocessing_distributed"]:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            config["rank"] = config["rank"] * ngpus_per_node + gpu
        dist.init_process_group(
            backend=config["dist_backend"],
            init_method=config["dist_url"],
            world_size=config["world_size"],
            rank=config["rank"],
        )

    # Disable logging on all workers except rank=0
    if config["multiprocessing_distributed"] and config["rank"] % ngpus_per_node != 0:
        logging.basicConfig(
            format="%(asctime)s %(message)s", level=logging.WARNING, force=True
        )

    model = load_model(gpu=config["gpu"], config=config["teacher"])
    if config["gpu"] is not None:
        torch.cuda.set_device(config["gpu"])
        model = model.cuda(config["gpu"])
    else:
        model.cuda()

    # Set model to eval
    model.eval()

    # Print number of model parameters
    logging.info(
        "Number of model parameters: {}".format(
            sum(p.numel() for p in model.parameters() if p.requires_grad)
        )
    )

    # Data loading code
    logging.info("Instantiating dataset.")
    config["image_augmentation"] = {"train": {}, "val": {}}
    train_dataset, _ = get_dataset(config)
    num_samples = config["reinforce"]["num_samples"]
    # Set transforms to None and wrap the dataset
    train_dataset.transform = None
    tr = T_dr.compose_from_config(
        T_dr.before_collate_config(config["reinforce"]["image_augmentation"])
    )
    train_dataset = DatasetWithParameters(
        train_dataset, transform=tr, num_samples=num_samples
    )

    train_dataset = IndexedDataset(train_dataset)
    if config["distributed"]:
        # When using a single GPU per process and per
        # DistributedDataParallel, we need to divide the batch size
        # ourselves based on the total number of GPUs we have
        config["batch_size"] = int(config["batch_size"] / ngpus_per_node)
        config["workers"] = int(
            (config["workers"] + ngpus_per_node - 1) / ngpus_per_node
        )
        # Shuffle=True is better for mixing reinforcements
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, shuffle=True, drop_last=False
        )
        train_sampler.set_epoch(0)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=(train_sampler is None),
        drop_last=False,
        num_workers=config["workers"],
        pin_memory=config["pin_memory"],
        sampler=train_sampler,
    )

    reinforce(train_loader, model, config)


def reinforce(
    train_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    config: Dict[str, Any],
) -> None:
    """Generate reinforcements and save in individual files per training sample."""
    batch_time = AverageMeter("Time", ":6.3f")
    progress = ProgressMeter(len(train_loader), [batch_time])

    num_samples = config["reinforce"]["num_samples"]
    transforms = T_dr.compose_from_config(config["reinforce"]["image_augmentation"])

    rdata_ext_path = os.path.join("/mnt/reinforced_data")
    os.makedirs(rdata_ext_path, exist_ok=True)

    with torch.no_grad():
        end = time.time()
        for batch_i, data in enumerate(train_loader):
            ids, images, target, coords = data
            if config["gpu"] is not None:
                images = images.cuda(config["gpu"], non_blocking=True)
                target = target.cuda(config["gpu"], non_blocking=True)

            # Apply transformations
            images_aug, params_aug = transform_batch(
                ids, images, target, coords, transforms, config
            )
            images_aug = images_aug.reshape((-1,) + images.shape[2:])

            # Compute output
            prob = model(images_aug, return_prob=True)
            prob = prob.reshape((images.shape[0], num_samples, -1))
            prob = prob.cpu().numpy()
            for j in range(images.shape[0]):
                new_samples = [
                    {
                        "params": params_aug[j][k],
                        "prob": sparsify(prob[j][k], config["reinforce"]["topk"]),
                    }
                    for k in range(num_samples)
                ]
                fname = os.path.join(rdata_ext_path, "{}.pth.tar".format(int(ids[j])))
                if not os.path.exists(fname):
                    if not config["reinforce"]["joblib"]:
                        # protocol=4 gives ~1.3x compression vs proto=1
                        torch.save((int(ids[j]), new_samples), fname, pickle_protocol=4)
                    else:
                        # joblib gives ~2.2x compression vs torch.save(proto=4)
                        # but 1.6x slower to save and 7x slower to load
                        joblib.dump((int(ids[j]), new_samples), fname, compress=True)

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if batch_i % config["print_freq"] == 0:
                progress.display(batch_i)

    time.sleep(0.01)  # Preventing broken pipe error


def transform_batch(
    ids: torch.Tensor,
    images: torch.Tensor,
    target: torch.Tensor,
    coords: torch.Tensor,
    transforms: T_dr.Compose,
    config: Dict[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply image transformations to a batch and return parameters.

    Args:
        ids: Image indices. Shape: [batch_size, 1]
        images: Image crops. If random-resized-crop is enabled, `imgaes` has
            multiple random crop.
            Shape without RRC: [batch_size, n_channels, crop_height, crop_width],
            Shape with RRC: [batch_size, n_samples, n_channels, crop_height, crop_width]
        target: Ground-truth labels. Shape: [batch_size, 1]
        coords: Random-resized-crop coordinates. Shape: [batch_size, n_samples, 4]
        transforms: A list of transformations to be applied randomly and stored.
        config: A dictionary of configurations with `reinforce` key.

    Returns:
        A tuple of transformed images and transformation parameters.
        Shape [0]: [batch_size, n_samples, n_channels, crop_height, crop_width]
        Shape [1]: A list of size [batch_size, n_samples].
    """
    num_samples = config["reinforce"]["num_samples"]
    images_aug = []
    params_aug = []
    for i in range(images.shape[0]):
        images_aug += [[]]
        params_aug += [[]]
        for j in range(num_samples):
            # Choose a sample with pre-collate transformation parameters
            img = images[i, j]

            # Sample an image pair
            index = random.randint(0, images.shape[0] - 1)
            j2 = random.randint(0, images.shape[1] - 1)
            img2 = images[index, j2]
            id2 = int(ids[index])

            # Apply remaining transformations after collate
            img, params = transforms(img, img2, after_collate=True)

            # Update parameters with before collate transformations
            params0 = {
                k: (tuple(v[i, j].tolist()), tuple(v[index, j2].tolist()))
                for k, v in coords.items()
            }
            params.update(params0)

            if "mixup" in params or "cutmix" in params:
                params["id2"] = id2
            if config["reinforce"]["compress"]:
                params = transforms.compress(params)
            images_aug[-1] += [img]
            params_aug[-1] += [params]
        images_aug[-1] = torch.stack(images_aug[-1], axis=0)
    images_aug = torch.stack(images_aug, axis=0)
    return images_aug, params_aug


if __name__ == "__main__":
    args = parse_args()
    main(args)
