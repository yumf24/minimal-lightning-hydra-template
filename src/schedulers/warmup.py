"""Custom learning rate schedulers with warmup support.

This module provides LR schedulers that combine warmup with various annealing
strategies, commonly used in deep learning training pipelines.
"""

import math
from typing import List

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class WarmupCosineScheduler(LRScheduler):
    """Linear warmup followed by cosine annealing.

    This scheduler gradually increases the learning rate from 0 to the target
    value during warmup, then applies cosine annealing to decrease it.

    Args:
        optimizer: Wrapped optimizer
        warmup_steps: Number of warmup steps
        max_steps: Total number of training steps
        eta_min: Minimum learning rate after annealing (default: 0)
        last_epoch: The index of last epoch (default: -1)

    Example:
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        >>> scheduler = WarmupCosineScheduler(optimizer, warmup_steps=1000, max_steps=10000)
        >>> for step in range(max_steps):
        ...     optimizer.step()
        ...     scheduler.step()
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        max_steps: int,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """Compute learning rate for current step."""
        if self.last_epoch < self.warmup_steps:
            # Linear warmup: lr = base_lr * (step / warmup_steps)
            return [
                base_lr * (self.last_epoch + 1) / self.warmup_steps
                for base_lr in self.base_lrs
            ]
        # Cosine annealing after warmup
        progress = (self.last_epoch - self.warmup_steps) / (self.max_steps - self.warmup_steps)
        cos_value = math.cos(progress * math.pi)
        return [
            self.eta_min + (base_lr - self.eta_min) * (1 + cos_value) / 2
            for base_lr in self.base_lrs
        ]


class WarmupLinearScheduler(LRScheduler):
    """Linear warmup followed by linear decay.

    Args:
        optimizer: Wrapped optimizer
        warmup_steps: Number of warmup steps
        max_steps: Total number of training steps
        eta_min: Minimum learning rate after decay (default: 0)
        last_epoch: The index of last epoch (default: -1)
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        max_steps: int,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """Compute learning rate for current step."""
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            return [
                base_lr * (self.last_epoch + 1) / self.warmup_steps
                for base_lr in self.base_lrs
            ]
        # Linear decay after warmup
        progress = (self.last_epoch - self.warmup_steps) / (self.max_steps - self.warmup_steps)
        return [
            self.eta_min + (base_lr - self.eta_min) * (1 - progress)
            for base_lr in self.base_lrs
        ]


class WarmupConstantScheduler(LRScheduler):
    """Linear warmup followed by constant learning rate.

    Args:
        optimizer: Wrapped optimizer
        warmup_steps: Number of warmup steps
        last_epoch: The index of last epoch (default: -1)
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """Compute learning rate for current step."""
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            return [
                base_lr * (self.last_epoch + 1) / self.warmup_steps
                for base_lr in self.base_lrs
            ]
        # Constant after warmup
        return self.base_lrs