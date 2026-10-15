"""Two-stage cosine LR for the from-scratch harvest-tomato experiment."""
from __future__ import annotations

import math

from torch.optim.lr_scheduler import LambdaLR


class TwoStageCosineFactor:
    def __init__(self, *, base_lr: float, middle_lr: float, final_lr: float,
                 warmup_steps: int, first_stage_steps: int, total_steps: int):
        if not (0 < final_lr < middle_lr < base_lr):
            raise ValueError("Expected 0 < final_lr < middle_lr < base_lr")
        if not (0 < warmup_steps < first_stage_steps < total_steps):
            raise ValueError("Expected warmup < first-stage end < total steps")
        self.base_lr = base_lr
        self.middle_lr = middle_lr
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.first_stage_steps = first_stage_steps
        self.total_steps = total_steps

    def __call__(self, step: int) -> float:
        if step < self.warmup_steps:
            return step / self.warmup_steps
        if step <= self.first_stage_steps:
            progress = (step - self.warmup_steps) / (
                self.first_stage_steps - self.warmup_steps)
            lr = self.middle_lr + (self.base_lr - self.middle_lr) * (
                1 + math.cos(math.pi * progress)) / 2
        else:
            progress = min(1.0, (step - self.first_stage_steps) / (
                self.total_steps - self.first_stage_steps))
            lr = self.final_lr + (self.middle_lr - self.final_lr) * (
                1 + math.cos(math.pi * progress)) / 2
        return lr / self.base_lr


def get_harvest_two_stage_scheduler(optimizer, *, steps_per_epoch: int,
                                    total_epochs: int, first_stage_epochs: int,
                                    warmup_steps: int, middle_lr: float,
                                    final_lr: float):
    base_lrs = {group["lr"] for group in optimizer.param_groups}
    if len(base_lrs) != 1:
        raise ValueError("Harvest scheduler expects one shared base learning rate")
    factor = TwoStageCosineFactor(
        base_lr=base_lrs.pop(), middle_lr=middle_lr, final_lr=final_lr,
        warmup_steps=warmup_steps,
        first_stage_steps=steps_per_epoch * first_stage_epochs,
        total_steps=steps_per_epoch * total_epochs,
    )
    return LambdaLR(optimizer, lr_lambda=factor)
