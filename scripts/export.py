from __future__ import annotations

import argparse
from pathlib import Path

from void import export


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("ckpt", type=Path)
    p.add_argument("--format", choices=("gguf", "gptq"), required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--quant", default="Q4_K_M")
    args = p.parse_args()
    if args.format == "gguf":
        export.to_gguf(args.ckpt, args.out, args.quant)
    else:
        export.to_gptq(args.ckpt, args.out, [])


if __name__ == "__main__":
    main()
