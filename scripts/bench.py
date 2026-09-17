"""Microbenchmarks for the forward pass and full train step."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("config", type=Path)
    p.add_argument("--steps", type=int, default=50)
    p.parse_args()
    raise NotImplementedError


if __name__ == "__main__":
    main()
