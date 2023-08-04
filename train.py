#
# Copyright (C) 2023 Apple Inc. All rights reserved.
#
"""Modification of Pytorch ImageNet training code to handle additional datasets."""
import argparse
import os
import random
import shutil
import warnings
import yaml
import copy
from typing import List, Dict, Any

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed

from trainers import get_trainer
from utils import CosineLR
from data import (
    download_dataset,
    get_dataset,
    get_dataset_num_classes,
    get_dataset_size,
)

from dr.data import ReinforcedDataset

import logging

logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)

best_acc1 = 0


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="PyTorch Training")
    parser.add_argument(
        "--config", type=str, required=False, help="Path to a yaml config file."
    )
    args = parser.parse_args()
    return args


def get_trainable_parameters(
    model: torch.nn.Module,
    weight_decay: float,
    no_decay_bn_filter_bias: bool,
    *args,
    **kwargs
) -> List[Dict[str, List[torch.nn.Parameter]]]:
    """Return trainable model parameters excluding biases and normalization layers.

    Args:
        model: The Torch model to be trained.
        weight_decay: The weight decay coefficient.
        no_decay_bn_filter_bias: If True exclude biases and normalization layer params.

    Returns:
        A list with two dictionaries for parameters with and without weight decay.
    """
    with_decay = []
    without_decay = []
    for _, param in model.named_parameters():
        if param.requires_grad and len(param.shape) == 1 and no_decay_bn_filter_bias:
            # biases and normalization layer parameters are of len 1
            without_decay.append(param)
        elif param.requires_grad:
            with_decay.append(param)
    param_list = [{"params": with_decay, "weight_decay": weight_decay}]
    if len(without_decay) > 0:
        param_list.append({"params": without_decay, "weight_decay": 0.0})
    return param_list


def get_optimizer(
    model: torch.nn.Module, config: Dict[str, Any]
) -> torch.optim.Optimizer:
    """Initialize an optimizer with parameters of a model.

    Args:
        model: The model to be trained.
        config: A dictionary of optimizer hyperparameters and configurations. The
            configuration should at least have a `name`.

    Returns:
        A Torch optimizer.
    """
    optim_name = config["name"]

    params = get_trainable_parameters(
        model,
        weight_decay=config.get("weight_decay", 0.0),
        no_decay_bn_filter_bias=config.get("no_decay_bn_filter_bias", False),
    )

    if optim_name == "sgd":
        optimizer = torch.optim.SGD(
            params=params,
            lr=config["lr"],
            momentum=config["momentum"],
            nesterov=config.get("nesterov", True),
        )
    elif optim_name == "adamw":
        optimizer = torch.optim.AdamW(
            params=params,
            lr=config["lr"],
            betas=(config["beta1"], config["beta2"]),
        )
    else:
        raise NotImplementedError
    return optimizer


def main(args) -> None:
    """Train a model with the configurations specified in given arguments."""
    if args.config is not None:
        # Read parameters from yaml config for local run
        yaml_path = args.config
        with open(yaml_path, "r") as file:
            config = yaml.safe_load(file).get("parameters")

    dataset = config.get("dataset", "imagenet")
    config["num_classes"] = get_dataset_num_classes(dataset)

    # Print args before training
    logging.info("Training args: {}".format(config))

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


def main_worker(gpu: int, ngpus_per_node: int, config: Dict[str, Any]) -> None:
    """Train a model with a single process. In distributed training, run on one GPU."""
    global best_acc1
    config["gpu"] = gpu

    if config["gpu"] is not None:
        logging.info("Use GPU: {} for training".format(config["gpu"]))

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

    # Initialize a trainer class that handles different models of training such as
    # standard training (ERM), Knowledge distillation, and Dataset Reinforcement
    trainer = get_trainer(config)

    # Create the model to train. Also create and load the teacher model for KD
    model = trainer.get_model()

    # Print number of model parameters
    logging.info(
        "Number of model parameters: {}".format(
            sum(p.numel() for p in model.parameters() if p.requires_grad)
        )
    )

    # Define loss function (criterion) and optimizer
    criterion = trainer.get_criterion()
    logging.info("Criterion: {}".format(criterion))

    # Define optimizer
    optimizer = get_optimizer(model, config["optim"])
    logging.info("Optimizer: {}".format(optimizer))

    dataset_size = get_dataset_size(config["dataset"])
    # Compute warmup and total iterations on each gpu
    warmup_length = (
        config["optim"]["warmup_length"]
        * dataset_size
        // config["batch_size"]
        // ngpus_per_node
    )
    total_steps = (
        config["epochs"] * dataset_size // config["batch_size"] // ngpus_per_node
    )
    lr_scheduler = CosineLR(
        optimizer,
        warmup_length=warmup_length,
        total_steps=total_steps,
        lr=config["optim"]["lr"],
        end_lr=config["optim"].get("end_lr", 0.0),
    )

    # Resume from a checkpoint
    if config["resume"]:
        if os.path.isfile(config["resume"]):
            logging.info("=> loading checkpoint '{}'".format(config["resume"]))
            if config["gpu"] is None:
                checkpoint = torch.load(config["resume"])
            else:
                # Map model to be loaded to specified single gpu.
                loc = "cuda:{}".format(config["gpu"])
                checkpoint = torch.load(config["resume"], map_location=loc)
            config["start_epoch"] = checkpoint["epoch"]
            best_acc1 = checkpoint["best_acc1"]
            if hasattr(model, "module"):
                model.module.load_state_dict(checkpoint["state_dict"])
            else:
                model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["scheduler"])
            logging.info(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    config["resume"], checkpoint["epoch"]
                )
            )
        else:
            logging.info("=> no checkpoint found at '{}'".format(config["resume"]))

    # Data loading code
    logging.info("Instantiating dataset.")
    train_dataset, val_dataset = get_dataset(config)
    logging.info("Training Dataset: {}".format(train_dataset))
    logging.info("Validation Dataset: {}".format(val_dataset))

    if config["trainer"] == "DR":
        # Reinforce dataset
        train_dataset = ReinforcedDataset(
            train_dataset,
            rdata_path=config["reinforce"]["data_path"],
            config=config["reinforce"],
            num_classes=config["num_classes"],
        )

    if config["distributed"]:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, shuffle=False, drop_last=True
        )
    else:
        train_sampler = None
        val_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=(train_sampler is None),
        num_workers=config["workers"],
        pin_memory=config["pin_memory"],
        persistent_workers=config['persistent_workers'],
        sampler=train_sampler,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["workers"],
        pin_memory=config["pin_memory"],
        persistent_workers=config['persistent_workers'],
        sampler=val_sampler,
    )

    if config["evaluate"]:
        # Evaluate a pretrained model without training
        trainer.validate(val_loader, model, criterion, config)
        return

    # Evaluate a pretrained teacher before training
    trainer.validate_pretrained(val_loader, model, criterion, config)

    # Start training
    for epoch in range(config["start_epoch"], config["epochs"]):
        if config["distributed"]:
            train_sampler.set_epoch(epoch)

        # Train for one epoch
        train_metrics = trainer.train(
            train_loader,
            model,
            criterion,
            optimizer,
            epoch,
            config,
            lr_scheduler,
        )

        # Evaluate on validation set
        val_metrics = trainer.validate(val_loader, model, criterion, config)
        val_acc1 = val_metrics["val_accuracy@top1"]

        # remember best acc@1 and save checkpoint
        is_best = val_acc1 > best_acc1
        best_acc1 = max(val_acc1, best_acc1)

        if not config["multiprocessing_distributed"] or (
            config["multiprocessing_distributed"]
            and config["rank"] % ngpus_per_node == 0
        ):
            metrics = dict(train_metrics)
            metrics.update(val_metrics)
            if isinstance(model, torch.nn.DataParallel) or isinstance(
                model, torch.nn.parallel.DistributedDataParallel
            ):
                model_state_dict = model.module.state_dict()
            else:
                model_state_dict = model.state_dict()
            is_save_epoch = (epoch + 1) % config.get("save_freq", 1) == 0
            is_save_epoch = is_save_epoch or ((epoch + 1) == config["epochs"])
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "config": config,
                    "state_dict": model_state_dict,
                    "best_acc1": best_acc1,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": lr_scheduler.state_dict(),
                },
                is_best,
                is_save_epoch,
                config["artifact_path"],
            )


def save_checkpoint(state, is_best, is_save_epoch, artifact_path) -> None:
    """Save checkpoint and update the best model."""
    fname = os.path.join(artifact_path, "checkpoint.pth.tar")
    torch.save(state, fname)

    if is_save_epoch:
        checkpoint_fname = os.path.join(
            artifact_path, "checkpoint_{}.pth.tar".format(state["epoch"])
        )
        shutil.copyfile(fname, checkpoint_fname)
    if is_best:
        best_model_fname = os.path.join(artifact_path, "model_best.pth.tar")
        shutil.copyfile(fname, best_model_fname)


if __name__ == "__main__":
    args = parse_args()
    main(args)
