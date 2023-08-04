#
# Copyright (C) 2023 Apple Inc. All rights reserved.
#
"""Utilities for training."""
from typing import List, Tuple, Optional

import torch
import numpy as np


def sparsify(p: np.array, k: int) -> Tuple[List[float], List[float]]:
    """Sparsify a probabilities vector to k top values."""
    ids = np.argpartition(p, -k)[-k:]
    return p[ids].tolist(), ids.tolist()


def densify(
    sp: Tuple[List[float], List[float]],
    num_classes: int,
    method: Optional[str] = "smooth",
) -> torch.Tensor:
    """Densify a sparse probability vector."""
    if not isinstance(sp, list) and not isinstance(sp, tuple):
        return torch.tensor(sp)
    sp, ids = torch.tensor(sp[0]), sp[1]
    r = 1.0 - sp.sum()  # Max with 0 is needed if sp.sum is close to 1.0
    # assert r >= 0.0, f"Sum of sparse probabilities ({r}) should be less than 1.0."
    if method == "zeros" or r < 0.0:
        p = torch.zeros(num_classes)
        p[ids] = torch.nn.functional.softmax(sp, dim=0)
    elif method == "smooth":
        p = torch.ones(num_classes) * r / (num_classes - len(sp))
        p[ids] = sp
    return p
