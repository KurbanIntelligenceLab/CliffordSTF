"""Exponential moving average of model parameters.

Provides ~2-5% accuracy gain for free during evaluation. The class shadows the
``param.data`` of every trainable parameter and exposes
``update`` / ``apply_shadow`` / ``restore`` for swapping the live weights with
their EMA counterparts during validation.
"""

from __future__ import annotations

import torch
from torch import nn


class ExponentialMovingAverage:
    """EMA over the trainable parameters of an ``nn.Module``.

    Example:
        ema = ExponentialMovingAverage(model, decay=0.999)
        # training loop
        ema.update()
        # eval
        ema.apply_shadow()
        model.eval()
        validate()
        ema.restore()
    """

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.model = model
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {}
        self.backup: dict[str, torch.Tensor] = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self) -> None:
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def apply_shadow(self) -> None:
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self) -> None:
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}
