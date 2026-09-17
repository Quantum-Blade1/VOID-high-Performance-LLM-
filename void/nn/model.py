from __future__ import annotations

import torch
from torch import Tensor, nn

from void.config import ModelConfig
from void.nn.attention import Attention
from void.nn.mlp import RMSNorm, SwiGLU, precompute_freqs_cis
from void.nn.mtp import MTPHead


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.attn = Attention(cfg, layer_idx)
        self.mlp = SwiGLU(cfg.dim, cfg.ffn_dim)
        self.attn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.mlp_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.window = cfg.sw_size if cfg.attn_pattern and cfg.attn_pattern[layer_idx] == "sw" else None

    def forward(self, x: Tensor, freqs_cis: Tensor, mask: Tensor | None) -> Tensor:
        x = x + self.attn(self.attn_norm(x), freqs_cis, mask)
        x = x + self.mlp(self.mlp_norm(x))
        return x


class VoidLM(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.registers = nn.Parameter(torch.zeros(cfg.n_registers, cfg.dim))
        self.layers = nn.ModuleList(Block(cfg, i) for i in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_embed.weight
        self.mtp_heads = nn.ModuleList(
            MTPHead(cfg, offset=i + 2) for i in range(cfg.n_mtp_heads)
        )
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta),
            persistent=False,
        )
        self._init_weights()

    def _init_weights(self) -> None:
        raise NotImplementedError

    def forward(self, tokens: Tensor) -> dict[str, Tensor]:
        """Returns {"logits": [B, T, V], "mtp_logits": [K, B, T, V]}."""
        raise NotImplementedError

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_embed.weight.numel()
        return n
