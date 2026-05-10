"""Global seeding utility.

Used everywhere a stochastic operation (numpy sampling, torch initialization,
GP variational optimization) is performed. Required by the reproducibility
rules in ``CLAUDE.md``.

This module does not implement any equation from the paper; it exists only
to make experiments deterministic across runs.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """Seed ``random``, ``numpy``, and ``torch`` (CPU + CUDA if available).

    Parameters
    ----------
    seed : int
        Non-negative integer seed.

    Notes
    -----
    Also sets ``PYTHONHASHSEED`` for hash-based determinism in spawned
    subprocesses and toggles cuDNN to its deterministic algorithms.
    Reproducibility rule mandated by ``CLAUDE.md``.
    """
    if seed is None or int(seed) < 0:
        raise ValueError(f"seed must be a non-negative integer, got {seed!r}")
    seed = int(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # cuDNN determinism (no-op on CPU-only installs, harmless otherwise).
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
