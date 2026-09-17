from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Source:
    hf_repo: str
    weight: float
    subset: str | None = None


@dataclass
class Sample:
    text: str
    source: str
    meta: dict[str, object]


def stream(sources: list[Source]) -> Iterator[Sample]:
    raise NotImplementedError


def quality_filter(samples: Iterable[Sample]) -> Iterator[Sample]:
    """Language ID + heuristic quality; source-specific thresholds set inline."""
    raise NotImplementedError


def dedup(samples: Iterable[Sample], min_hash_bands: int = 20) -> Iterator[Sample]:
    raise NotImplementedError


def scrub_contamination(samples: Iterable[Sample], banned: set[str]) -> Iterator[Sample]:
    raise NotImplementedError


def apply_fim(samples: Iterable[Sample], p: float = 0.5) -> Iterator[Sample]:
    """Reformat p fraction of code samples as prefix/suffix/middle."""
    raise NotImplementedError


def pack_repos(samples: Iterable[Sample], seq_len: int) -> Iterator[Sample]:
    """Toposort files within a repo and glue them into single sequences."""
    raise NotImplementedError


def write_shards(samples: Iterable[Sample], out_dir: Path, shard_tokens: int) -> None:
    raise NotImplementedError
