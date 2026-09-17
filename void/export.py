from __future__ import annotations

from pathlib import Path


def to_gguf(ckpt: Path, out: Path, quantize: str = "Q4_K_M") -> None:
    """Delegates to llama.cpp's convert.py; caller must have it installed."""
    raise NotImplementedError


def to_gptq(ckpt: Path, out: Path, calibration_shards: list[Path]) -> None:
    raise NotImplementedError


def write_model_card(ckpt: Path, eval_results: list, out: Path) -> None:
    raise NotImplementedError
