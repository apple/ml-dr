#
# Copyright (C) 2023 Apple Inc. All rights reserved.
#

"""Methods to create, load, and ensemble models."""
import os
from typing import Any, Dict, Generator, List, Optional, Tuple, Union
import yaml
import logging
import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.models as models



def move_to_device(model: torch.nn.Module, config: Dict[str, Any]) -> torch.nn.Module:
    """Wrap model with DDP/DP if distributed, convert to CUDA if GPU set, else CPU."""
    if not torch.cuda.is_available():
        logging.info("using CPU, this will be slow")
    elif config["distributed"]:
        ngpus_per_node = torch.cuda.device_count()
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if config["gpu"] is not None:
            torch.cuda.set_device(config["gpu"])
            model.cuda(config["gpu"])
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            config["batch_size"] = int(config["batch_size"] / ngpus_per_node)
            config["workers"] = int(
                (config["workers"] + ngpus_per_node - 1) / ngpus_per_node
            )
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[config["gpu"]]
            )
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to
            # all available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif config["gpu"] is not None:
        torch.cuda.set_device(config["gpu"])
        model = model.cuda(config["gpu"])
    else:
        # DataParallel will divide and allocate batch_size to all available
        # GPUs
        model = torch.nn.DataParallel(model).cuda()
    return model


def load_model(gpu: torch.device, config: Dict[str, Any]) -> torch.nn.Module:
    """Load a pretrained model or an ensemble of pretrained models."""
    if config.get("ensemble", False):
        # Load an ensemble from a checkpoint path
            config
        device = None
        if gpu is not None:
            device = "cuda:{}".format(gpu)
        # Load models
        members = torch.nn.ModuleList(load_ensemble(checkpoints_path, device))
        model = ClassificationEnsembleNet(members)
    elif config.get("timm_ensemble", False):
        import timm
        # Load an ensemble of Timm models
        model_names = config.get("name", None)
        if gpu is not None:
            torch.cuda.set_device(gpu)
        # Load pretrained models
        members = torch.nn.ModuleList()
        if not isinstance(model_names, list):
            model_names = model_names.split(",")
        for m in model_names:
            members += [timm.create_model(m, pretrained=True)]
        # Create Ensemble
        model = ClassificationEnsembleNet(members)
    elif config.get("checkpoint", None) is not None:
        # Load a single pretrained model
        checkpoint_path = config["checkpoint"]
        arch = config["arch"]
        model = load_from_local(checkpoint_path, arch, gpu)
    else:
        # Use default pretrained model from pytorch.
        model = models.__dict__[config["arch"]](pretrained=True)
    return model


def load_from_local(checkpoint_path: str, arch: str, gpu: int) -> torch.nn.Module:
    """Load model from local path and move to GPU if gpu set."""
    teacher_model = models.__dict__[arch]()

    # Load from checkpoint
    if gpu is None:
        checkpoint = torch.load(checkpoint_path)
    else:
        # Map model to be loaded to specified single gpu.
        loc = "cuda:{}".format(gpu)
        checkpoint = torch.load(checkpoint_path, map_location=loc)

    # Strip module. from checkpoint
    ckpt = {k.replace("module.", ""): v for k, v in checkpoint["state_dict"].items()}
    teacher_model.load_state_dict(ckpt)
    logging.info("Loaded checkpoint {} for teacher".format(checkpoint_path))
    return teacher_model


def create_model(arch: str, config: dict) -> torch.nn.Module:
    """Create models from CVNets/Timm/Torch."""
    if arch == "cvnets":
        import cvnets
        from cvnets import modeling_arguments
        import argparse
        # TODO: cvnets does not yet support easy model creation outside the library
        parser = argparse.ArgumentParser(description="")
        parser = modeling_arguments(parser)
        opts = parser.parse_args()
        config_dot = dict(convert_dict_to_dotted(config))
        for k, v in config_dot.items():
            if hasattr(opts, k):
                setattr(opts, k, v)
        setattr(opts, 'dataset.category', 'classification')
        model = cvnets.get_model(config_dot)
    elif arch == "timm":
        import timm
        model = timm.create_model(**config["model"])
    else:
        model = models.__dict__[arch]()
    logging.info(model)
    return model


class ClassificationEnsembleNet(nn.Module):
    """Ensemble model for classification based on averaging."""

    def __init__(self, members: torch.nn.ModuleList) -> None:
        """Init ensemble."""
        super().__init__()
        self.members = members

    def forward(
        self, x: torch.Tensor, return_prob: bool = False, temperature: float = 1.0
    ) -> torch.Tensor:
        """Reduce function for classification using averaging."""
        output = 0
        for a_network in self.members:
            logits = a_network(x)
            prob = F.softmax(logits / temperature, dim=1)
            output = output + prob

        if return_prob:
            return output / float(len(self.members))
        return (output / float(len(self.members))).log()


def init_model_from_ckpt(
    model: torch.nn.Module,
    ckpt_path: str,
    device: torch.device,
    strict_keys: Optional[bool] = False,
) -> torch.nn.Module:
    """Init a model from an already trained model.

    Args:
        model: the pytorch model object to be loaded.
        ckpt_path: path to a model checkpoint.
        device: the device to load data to. Note that
            the model could be saved from a different device.
            Here we transfer the paramters to the current given device. So,
            a model could be trained and saved on GPU, and be loaded on CPU,
            for example.
        strict_keys: If True keys in state_dict of both models should be
            identical.
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    pretrained_dict = ckpt["state_dict"]
    # For incorrectly saved DataParallel models
    pretrained_dict = {k.replace("module.", ""): v for k, v in pretrained_dict.items()}
    model.load_state_dict(pretrained_dict, strict=strict_keys)

    return model


def load_ensemble(checkpoints_path: str, device: torch.device) -> List[torch.nn.Module]:
    """Traverse all subdirs and load checkpoints."""
    models = list()
    for root, dirs, files in os.walk(checkpoints_path):
        dirs.sort()
        # a directory is a legitimate checkpoint directory if the root has a config.yaml
        if "config.yaml" in files:
            with open(os.path.join(root, "config.yaml")) as f:
                model_config = yaml.safe_load(f).get("parameters")
            arch = model_config["arch"]
            if (
                model_config.get("model", {})
                .get("classification", {})
                .get("pretrained", None)
                is not None
            ):
                model_config["model"]["classification"]["pretrained"] = None
            model = create_model(arch, model_config)
            ckpt_path = get_path_to_checkpoint(root)
            model = init_model_from_ckpt(model, ckpt_path, device)
            models.append(model)
    return models


def get_path_to_checkpoint(artifact_dir: str, epoch=None) -> str:
    """Find checkpoint file path in an artifact directory.

    Args:
        artifact_dir: path to an experiment artifact directory,
            to laod checkpoints from there.
        epoch: If given tries to load that checkpoint, otherwise
            loads the latest. This function assumes checkpoints are saved
            as `checkpoint_epoch.tar'
    """
    ckpts_path = os.path.join(artifact_dir, "checkpoints")
    ckpts_list = os.listdir(ckpts_path)
    ckpts_dict = {
        int(ckpt.split("_")[1].split(".")[0]): os.path.join(ckpts_path, ckpt)
        for ckpt in ckpts_list
    }
    if len(ckpts_list) == 0:
        msg = "No checkpoint exists!"
        raise ValueError(msg)
    if epoch is not None:
        if epoch not in ckpts_dict.keys():
            msg = "Could not find checkpoint for epoch {} !"
            raise ValueError(msg.format(epoch))
    else:
        epoch = max(ckpts_dict.keys())
    return ckpts_dict[epoch]


def convert_dict_to_dotted(
    c: Dict[str, Any],
    prefix: str = "",
) -> Generator[Union[Any, Tuple[str, Any]], None, None]:
    """Convert a nested dictionary of configs to flat dotted notation."""
    if isinstance(c, dict):
        prefix += "." if prefix != "" else ""
        for k, v in c.items():
            for x in convert_dict_to_dotted(v, prefix + k):
                yield x
    else:
        yield (prefix, c)
