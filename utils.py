#
# Copyright (C) 2023 Apple Inc. All rights reserved.
#
"""Utilities for training."""
from enum import Enum
from typing import Dict, Any, Iterable, List, Optional

import torch
from torch import Tensor
import numpy as np
import logging

import torch.distributed as dist


class Summary(Enum):
    """Meter value types."""

    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ""
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.3f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.3f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.3f}"
        else:
            raise ValueError("invalid summary type %r" % self.summary_type)

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logging.info("\t".join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        logging.info(" ".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def accuracy(
    output: Tensor, target: Tensor, topk: Optional[Iterable[int]] = (1,)
) -> List[float]:
    """Compute the accuracy over the k top predictions for the specified values of k."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        if len(target.shape) > 1 and target.shape[1] > 1:
            # soft labels
            _, target = target.max(dim=1)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res


def assign_learning_rate(optimizer: torch.optim.Optimizer, new_lr: float) -> None:
    """Update lr parameter of an optimizer.

    Args:
        optimizer: A torch optimizer.
        new_lr: updated value of learning rate.
    """
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


def _warmup_lr(base_lr: float, warmup_length: int, n_iter: int) -> float:
    """Get updated lr after applying initial warmup.

    Args:
        base_lr: Nominal learning rate.
        warmup_length: Number of total iterations for initial warmup.
        n_iter: Current iteration number.

    Returns:
        Warmup-updated learning rate.
    """
    return base_lr * (n_iter + 1) / warmup_length


class CosineLR:
    """LR adjustment callable with cosine schedule.

    Args:
        optimizer: A torch optimizer.
        warmup_length: Number of iterations for initial warmup.
        total_steps: Total number of iterations.
        lr: Nominal learning rate value.

    Returns:
        A callable to adjust learning rate per iteration.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_length: int,
        total_steps: int,
        lr: float,
        end_lr: float = 0.0,
        **kwargs
    ) -> None:
        """Set parameters of cosine learning rate with warmup."""
        assert lr > end_lr, (
            "End LR should be less than the LR. Got:" " lr={} and last_lr={}"
        ).format(lr, end_lr)
        self.optimizer = optimizer
        self.warmup_length = warmup_length
        self.total_steps = total_steps
        self.lr = lr
        self.last_lr = 0
        self.end_lr = end_lr
        self.last_n_iter = 0

    def step(self) -> float:
        """Return updated learning rate for the next iteration."""
        self.last_n_iter += 1
        n_iter = self.last_n_iter

        if n_iter < self.warmup_length:
            new_lr = _warmup_lr(self.lr, self.warmup_length, n_iter)
        else:
            e = n_iter - self.warmup_length + 1
            es = self.total_steps - self.warmup_length

            new_lr = self.end_lr + 0.5 * (self.lr - self.end_lr) * (
                1 + np.cos(np.pi * e / es)
            )

        assign_learning_rate(self.optimizer, new_lr)

        self.last_lr = new_lr

    def get_last_lr(self) -> List[float]:
        """Return the value of the last learning rate."""
        return [self.last_lr]

    def state_dict(self) -> Dict[str, Any]:
        """Return the state dictionary to recover optimization in training restart."""
        return {
            "warmup_length": self.warmup_length,
            "total_steps": self.total_steps,
            "lr": self.lr,
            "last_lr": self.last_lr,
            "last_n_iter": self.last_n_iter,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore scheduler state."""
        self.warmup_length = state["warmup_length"]
        self.total_steps = state["total_steps"]
        self.lr = state["lr"]
        self.last_lr = state["last_lr"]
        self.last_n_iter = state["last_n_iter"]
