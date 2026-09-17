"""End-to-end corpus build: filter → dedup → contam → FIM → pack → tokenize → shard."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    tok = sub.add_parser("tokenizer", help="train BPE")
    tok.add_argument("--corpus-glob", required=True)
    tok.add_argument("--vocab", type=int, default=49_152)
    tok.add_argument("--out", type=Path, required=True)

    prep = sub.add_parser("prepare", help="run the corpus pipeline")
    prep.add_argument("--mix", required=True, help="data mix YAML")
    prep.add_argument("--out-repo", required=True)

    args = p.parse_args()
    raise NotImplementedError(args.cmd)


if __name__ == "__main__":
    main()
