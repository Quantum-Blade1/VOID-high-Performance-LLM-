from __future__ import annotations

from pathlib import Path

FIM_TOKENS = ("<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>")
CHATML_TOKENS = ("<|im_start|>", "<|im_end|>")
SPECIAL_TOKENS = ("<|endoftext|>", *FIM_TOKENS, *CHATML_TOKENS)


def train_bpe(corpus_glob: str, vocab_size: int, out: Path) -> None:
    raise NotImplementedError


def load(path: str | Path):  # returns transformers.PreTrainedTokenizerFast
    raise NotImplementedError
