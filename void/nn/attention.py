from __future__ import annotations

from math import exp

import torch
from torch import Tensor, nn

from void.config import ModelConfig
from void.nn.mlp import RMSNorm, apply_rotary


class Attention(nn.Module):
    """GQA with an optional second Q/K pair for differential attention."""

    def __init__(self, cfg: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.n_rep = cfg.n_heads // cfg.n_kv_heads
        self.diff = cfg.attn_kind == "diff-gqa"

        d, hd, kvd = cfg.dim, cfg.n_heads * cfg.head_dim, cfg.n_kv_heads * cfg.head_dim
        self.wq = nn.Linear(d, hd, bias=False)
        self.wk = nn.Linear(d, kvd, bias=False)
        self.wv = nn.Linear(d, kvd, bias=False)
        self.wo = nn.Linear(hd, d, bias=False)
        if self.diff:
            self.wq2 = nn.Linear(d, hd, bias=False)
            self.wk2 = nn.Linear(d, kvd, bias=False)
            if not cfg.tie_v:
                self.wv2 = nn.Linear(d, kvd, bias=False)
            self.lambda_q1 = nn.Parameter(torch.zeros(cfg.head_dim))
            self.lambda_k1 = nn.Parameter(torch.zeros(cfg.head_dim))
            self.lambda_q2 = nn.Parameter(torch.zeros(cfg.head_dim))
            self.lambda_k2 = nn.Parameter(torch.zeros(cfg.head_dim))
            self.lambda_init = cfg.lambda_init_base - cfg.lambda_init_decay * exp(
                -0.3 * (layer_idx - 1)
            )
        if cfg.qk_norm:
            self.q_norm = RMSNorm(cfg.head_dim, cfg.norm_eps)
            self.k_norm = RMSNorm(cfg.head_dim, cfg.norm_eps)

    def forward(
        self,
        x: Tensor,               # [B, T, D]
        freqs_cis: Tensor,       # [T, head_dim/2]
        mask: Tensor | None,
    ) -> Tensor:
        raise NotImplementedError
