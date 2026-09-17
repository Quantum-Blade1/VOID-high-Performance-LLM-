from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Result:
    name: str
    metric: str
    value: float
    n_examples: int


def humaneval(ckpt: Path, batch: int = 32) -> Result:
    raise NotImplementedError


def mbpp(ckpt: Path, batch: int = 32) -> Result:
    raise NotImplementedError


def lm_eval(ckpt: Path, tasks: list[str]) -> list[Result]:
    raise NotImplementedError


def perplexity(ckpt: Path, shard_glob: str) -> Result:
    raise NotImplementedError


def void_cli_bench(ckpt: Path) -> Result:
    """CLI reasoning benchmark drawn from fish/zsh/rg/fzf/bat/exa codebases."""
    raise NotImplementedError


def run_all(ckpt: Path, tasks: list[str]) -> list[Result]:
    raise NotImplementedError
