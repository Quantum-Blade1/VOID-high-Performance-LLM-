from __future__ import annotations

import argparse

from void.config import Config
from void.train.loop import train


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("config", type=str)
    p.add_argument("--resume", type=str, default=None)
    args = p.parse_args()
    train(Config.from_yaml(args.config), resume_from=args.resume)


if __name__ == "__main__":
    main()
