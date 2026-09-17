from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class TrainState:
    step: int
    tokens_seen: int
    shard: str
    shard_offset: int
    rng: dict[str, Any]


def save(state: TrainState, model, optimizers, path: Path) -> None:  # noqa: ANN001
    raise NotImplementedError


def load(path: Path) -> tuple[TrainState, dict[str, Any], list[dict[str, Any]]]:
    raise NotImplementedError


def push_to_hub(local: Path, repo: str, private: bool = True) -> None:
    raise NotImplementedError


def latest_from_hub(repo: str) -> Path | None:
    raise NotImplementedError
