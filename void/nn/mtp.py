from __future__ import annotations

from torch import Tensor, nn

from void.config import ModelConfig


class MTPHead(nn.Module):
    """Auxiliary head predicting a future token. One extra transformer layer + shared LM head."""

    def __init__(self, cfg: ModelConfig, offset: int) -> None:
        super().__init__()
        self.offset = offset
        # single block wired to the LM head in VoidLM
        raise NotImplementedError

    def forward(self, h: Tensor) -> Tensor:
        raise NotImplementedError
