from __future__ import annotations

import math
from collections.abc import Iterable

import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from void.config import OptimConfig


def split_params(model: nn.Module) -> tuple[list[Tensor], list[Tensor]]:
    """Return (hidden_matrices, everything_else). Hidden = 2D+ params outside
    embeddings, LM head, and norms."""
    hidden: list[Tensor] = []
    other: list[Tensor] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_hidden = (
            p.ndim >= 2
            and "embed" not in name
            and "lm_head" not in name
            and "norm" not in name
        )
        (hidden if is_hidden else other).append(p)
    return hidden, other


def build_optimizers(model: nn.Module, cfg: OptimConfig) -> list[Optimizer]:
    raise NotImplementedError


class Muon(Optimizer):
    """Newton-Schulz orthogonalised momentum. Hidden 2D+ params only."""

    def __init__(self, params: Iterable[Tensor], lr: float, momentum: float = 0.95, ns_iters: int = 5) -> None:
        super().__init__(params, dict(lr=lr, momentum=momentum, ns_iters=ns_iters))

    @torch.no_grad()
    def step(self, closure=None):  # noqa: ANN001
        raise NotImplementedError


def cosine_schedule(step: int, peak_lr: float, cfg: OptimConfig, total_steps: int) -> float:
    if step < cfg.warmup_steps:
        return peak_lr * step / max(1, cfg.warmup_steps)
    progress = (step - cfg.warmup_steps) / max(1, total_steps - cfg.warmup_steps)
    min_lr = peak_lr * cfg.min_lr_frac
    return min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + math.cos(math.pi * progress))
