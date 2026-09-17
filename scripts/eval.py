from __future__ import annotations

import argparse
from pathlib import Path

from void.eval import run_all


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("ckpt", type=Path)
    p.add_argument("--tasks", nargs="+", default=["humaneval", "mbpp", "mmlu", "hellaswag"])
    args = p.parse_args()
    for r in run_all(args.ckpt, args.tasks):
        print(f"{r.name:20s} {r.metric:10s} {r.value:.4f}  n={r.n_examples}")


if __name__ == "__main__":
    main()
