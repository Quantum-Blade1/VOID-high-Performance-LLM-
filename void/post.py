from __future__ import annotations

from pathlib import Path

from void.config import DPOConfig, SFTConfig


def sft(base_ckpt: Path, cfg: SFTConfig, out: Path) -> None:
    raise NotImplementedError


def dpo(sft_ckpt: Path, cfg: DPOConfig, out: Path) -> None:
    raise NotImplementedError


def soup(ckpts: list[Path], out: Path) -> None:
    """Uniform average of state dicts. All checkpoints must have identical keys."""
    raise NotImplementedError
