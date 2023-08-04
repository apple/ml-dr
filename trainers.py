#
# Copyright (C) 2023 Apple Inc. All rights reserved.
#
"""Training methods for ERM, Knowledge Distillation, and Dataset Reinforcement."""
from abc import ABC
import time
import logging
from typing import Callable, Dict, Any, Type

import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.nn import functional as F

from utils import AverageMeter, CosineLR, ProgressMeter, Summary, accuracy
from models import move_to_device, load_model, create_model
from transforms import MixingTransforms


class Trainer(ABC):
    """Abstract class for various training methodologies."""

    def get_model(self) -> nn.Module:
        """Create and initialize the model to train using self.config."""
        raise NotImplementedError("Implement `get_model` to initialize a model.")

    def get_criterion(self) -> nn.Module:
        """Return the training criterion."""
        raise NotImplementedError("Implement `get_criterion`.")

    def train(
        self,
        train_loader: DataLoader,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        config: Dict[str, Any],
        lr_scheduler: Type[CosineLR],
    ) -> Dict[str, Any]:
        """Train a model for a single epoch and return training metrics dictionary."""
        raise NotImplementedError("Implement `train` method.")

    def validate_pretrained(self, *args, **kwargs) -> None:
        """Validate pretrained teacher model."""
        pass

    def validate(self, *args, **kwargs) -> Dict[str, Any]:
        """Validate the model that is being trained and return a metrics dictionary."""
        return validate(*args, **kwargs)


def get_trainer(config: Dict[str, Any]) -> Trainer:
    """Initialize a trainer given a configuration dictionary."""
    trainer_type = config["trainer"]
    if trainer_type == "ERM":
        return ERMTrainer(config)
    elif trainer_type == "KD":
        return KDTrainer(config)
    elif trainer_type == "DR":
        return ReinforcedTrainer(config)
    raise NotImplementedError("Trainer not implemented.")


class ERMTrainer(Trainer):
    """Trainer class for Empirical Risk Minimization (ERM) with cross-entropy."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize ERMTrainer."""
        self.config = config
        self.label_smoothing = config["loss"].get("label_smoothing", 0.0)

    def get_model(self) -> torch.nn.Module:
        """Create and initialize the model to train using self.config."""
        arch = self.config["arch"]
        model = create_model(arch, self.config)
        model = move_to_device(model, self.config)
        return model

    def get_criterion(self) -> torch.nn.Module:
        """Return the training criterion."""
        criterion = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing).cuda(
            self.config["gpu"]
        )
        return criterion

    def train(
        self,
        train_loader: DataLoader,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        config: Dict[str, Any],
        lr_scheduler: Type[CosineLR],
    ) -> Dict[str, Any]:
        """Train a model for a single epoch and return training metrics dictionary."""
        batch_time = AverageMeter("Time", ":6.3f")
        data_time = AverageMeter("Data", ":6.3f")
        losses = AverageMeter("Loss", ":.6f")
        top1 = AverageMeter("Acc@1", ":6.2f")
        top5 = AverageMeter("Acc@5", ":6.2f")
        lrs = AverageMeter("Lr", ":.4f")
        conf = AverageMeter("Confidence", ":.5f")
        progress = ProgressMeter(
            len(train_loader),
            [batch_time, data_time, losses, top1, top5, lrs, conf],
            prefix="Epoch: [{}]".format(epoch),
        )

        # switch to train mode
        model.train()

        mixing_transforms = MixingTransforms(
            config["image_augmentation"], config["num_classes"]
        )

        end = time.time()
        for i, (images, target) in enumerate(train_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            if config["gpu"] is not None:
                images = images.cuda(config["gpu"], non_blocking=True)
                target = target.cuda(config["gpu"], non_blocking=True)

            # apply mixup / cutmix
            mix_images, mix_target = mixing_transforms(images, target)

            # compute output
            output = model(mix_images)

            # classification loss
            loss = criterion(output, mix_target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1, images.size(0))
            top5.update(acc5, images.size(0))
            lrs.update(lr_scheduler.get_last_lr()[0])

            # measure confidence
            prob = torch.nn.functional.softmax(output, dim=1)
            conf.update(prob.max(1).values.mean().item(), images.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            lr_scheduler.step()

            if i % config["print_freq"] == 0:
                progress.display(i)

        if config["distributed"]:
            top1.all_reduce()
            top5.all_reduce()

        metrics = {
            "train_accuracy@top1": top1.avg,
            "train_accuracy@top5": top5.avg,
            "train_loss": losses.avg,
            "lr": lrs.avg,
            "train_confidence": conf.avg,
        }
        return metrics


class ReinforcedTrainer(ERMTrainer):
    """Trainer with a reinforced dataset. Same as ERM Trainer with KL loss."""

    def get_criterion(self) -> Callable[[Tensor, Tensor], Tensor]:
        """Return KL loss instead of cross-entropy."""
        return lambda output, target: F.kl_div(
            F.log_softmax(output, dim=1), target, reduction="batchmean"
        )


class KDTrainer(ERMTrainer):
    """Trainer for Knowledge Distillation."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize trainer and set hyperparameters of KD."""
        # Loss config
        self.lambda_kd = config["loss"].get("lambda_kd", 1.0)
        self.lambda_cls = config["loss"].get("lambda_cls", 0.0)
        self.temperature = config["loss"].get("temperature", 1.0)
        assert self.temperature > 0, "Softmax with temperature=0 is undefined."
        self.label_smoothing = config["loss"].get("label_smoothing", 0.0)

        self.config = config
        self.teacher_model = None

    def get_model(self) -> torch.nn.Module:
        """Create and initialize student and teacher models."""
        config = self.config

        # Instantiate student model for training.
        student_arch = config["student"]["arch"]
        model = create_model(student_arch, config["student"])
        model = move_to_device(model, self.config)

        # Instantiate teacher model
        teacher_model = load_model(config["gpu"], config["teacher"])

        if config["gpu"] is not None:
            torch.cuda.set_device(config["gpu"])
            teacher_model = teacher_model.cuda(config["gpu"])
        else:
            teacher_model.cuda()
        # Set teacher to eval mode
        teacher_model.eval()
        self.teacher_model = teacher_model

        return model

    def validate_pretrained(
        self,
        val_loader: DataLoader,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        config: Dict[str, Any],
    ) -> None:
        """Validate teacher accuracy before training."""
        teacher_model = self.teacher_model
        do_validate = config.get("teacher", {}).get("validate", True)
        if teacher_model is not None and do_validate:
            logging.info(
                "Validation loader resizes to standard 256x256 resolution"
                " which is necessarily the optimal resolution for the teacher."
            )
            val_metrics = validate(val_loader, teacher_model, criterion, config)
            logging.info(
                "Teacher accuracy@top1: {}, @top5: {}".format(
                    val_metrics["val_accuracy@top1"], val_metrics["val_accuracy@top5"]
                )
            )

    def train(
        self,
        train_loader: DataLoader,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        config: Dict[str, Any],
        lr_scheduler: Type[CosineLR],
    ) -> Dict[str, Any]:
        """Train a model for a single epoch and return training metrics dictionary."""
        batch_time = AverageMeter("Time", ":6.3f")
        data_time = AverageMeter("Data", ":6.3f")
        losses = AverageMeter("Loss", ":.6f")
        kd_losses = AverageMeter("KD Loss", ":.6f")
        overall_losses = AverageMeter("Loss", ":.6f")
        top1 = AverageMeter("Acc@1", ":6.2f")
        top5 = AverageMeter("Acc@5", ":6.2f")
        lrs = AverageMeter("Lr", ":.4f")
        progress = ProgressMeter(
            len(train_loader),
            [batch_time, data_time, losses, kd_losses, overall_losses, top1, top5, lrs],
            prefix="Epoch: [{}]".format(epoch),
        )

        # Switch to train mode
        model.train()

        mixing_transforms = MixingTransforms(
            config["image_augmentation"], config["num_classes"]
        )

        end = time.time()
        for i, (images, target) in enumerate(train_loader):
            # Measure data loading time
            data_time.update(time.time() - end)

            if config["gpu"] is not None:
                images = images.cuda(config["gpu"], non_blocking=True)
                target = target.cuda(config["gpu"], non_blocking=True)

            # Apply mixup / cutmix
            mix_images, mix_target = mixing_transforms(images, target)

            # Compute output for differing resolution. Support only 224 student
            mix_images_small = mix_images
            if mix_images.shape[-1] != 224:
                mix_images_small = F.interpolate(
                    mix_images, size=(224, 224), mode="bilinear"
                )
            output = model(mix_images_small)

            # Classification loss
            loss = criterion(output, mix_target)
            losses.update(loss.item(), images.size(0))

            # Distillation loss
            # Get teacher's output for this input
            with torch.no_grad():
                teacher_probs = self.teacher_model(
                    mix_images, return_prob=True, temperature=self.temperature
                ).detach()
            kd_loss = F.kl_div(
                F.log_softmax(output / self.temperature, dim=1),
                teacher_probs,
                reduction="batchmean",
            ) * (self.temperature**2)
            kd_losses.update(kd_loss.item(), images.size(0))

            # Overall loss is a combination of kd loss and classification loss
            loss = self.lambda_cls * loss + self.lambda_kd * kd_loss

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            overall_losses.update(loss.item(), images.size(0))
            top1.update(acc1, images.size(0))
            top5.update(acc5, images.size(0))
            lrs.update(lr_scheduler.get_last_lr()[0])

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_scheduler.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % config["print_freq"] == 0:
                progress.display(i)

        if config["distributed"]:
            top1.all_reduce()
            top5.all_reduce()

        metrics = {
            "train_accuracy@top1": top1.avg,
            "train_accuracy@top5": top5.avg,
            "train_loss_ce": losses.avg,
            "train_loss_kd": kd_losses.avg,
            "train_loss_total": overall_losses.avg,
            "lr": lrs.avg,
        }
        return metrics


def validate(
    val_loader: DataLoader,
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Validate the model that is being trained and return a metrics dictionary."""

    def run_validate(loader: DataLoader, base_progress: int = 0) -> None:
        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(val_loader):
                i = base_progress + i
                if config["gpu"] is not None:
                    images = images.cuda(config["gpu"], non_blocking=True)
                    target = target.cuda(config["gpu"], non_blocking=True)

                # compute output
                output = model(images)
                # for validation, compute standard CE loss without label smoothing
                loss = F.cross_entropy(output, target)

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1, images.size(0))
                top5.update(acc5, images.size(0))

                # measure confidence
                prob = torch.nn.functional.softmax(output, dim=1)
                conf.update(prob.max(1).values.mean().item(), images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % config["print_freq"] == 0:
                    progress.display(i)

    batch_time = AverageMeter("Time", ":6.3f", Summary.NONE)
    losses = AverageMeter("Loss", ":.6f", Summary.NONE)
    top1 = AverageMeter("Acc@1", ":6.2f", Summary.AVERAGE)
    top5 = AverageMeter("Acc@5", ":6.2f", Summary.AVERAGE)
    conf = AverageMeter("Confidence", ":.5f", Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader)
        + (
            config["distributed"]
            and (
                len(val_loader.sampler) * config["world_size"] < len(val_loader.dataset)
            )
        ),
        [batch_time, losses, top1, top5],
        prefix="Test: ",
    )

    # switch to evaluate mode
    model.eval()

    # run validation using all nodes in a distributed env and aggregate results
    run_validate(val_loader)
    if config["distributed"]:
        top1.all_reduce()
        top5.all_reduce()

    if config["distributed"] and (
        len(val_loader.sampler) * config["world_size"] < len(val_loader.dataset)
    ):
        aux_val_dataset = Subset(
            val_loader.dataset,
            range(
                len(val_loader.sampler) * config["world_size"], len(val_loader.dataset)
            ),
        )
        aux_val_loader = torch.utils.data.DataLoader(
            aux_val_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=config["workers"],
            pin_memory=True,
        )
        run_validate(aux_val_loader, len(val_loader))

    progress.display_summary()

    metrics = {
        "val_loss": losses.avg,
        "val_accuracy@top1": top1.avg,
        "val_accuracy@top5": top5.avg,
        "val_confidence": conf.avg,
    }
    return metrics
