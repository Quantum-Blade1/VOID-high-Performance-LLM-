from __future__ import annotations

import pytest
import torch

from void.config import ModelConfig
from void.nn.mlp import RMSNorm, SwiGLU


def test_rmsnorm_shape_and_dtype():
    x = torch.randn(2, 8, 16, dtype=torch.bfloat16)
    y = RMSNorm(16)(x)
    assert y.shape == x.shape
    assert y.dtype == x.dtype


def test_swiglu_shape():
    x = torch.randn(2, 8, 16)
    assert SwiGLU(16, 32)(x).shape == x.shape


@pytest.mark.skip(reason="VoidLM.forward not implemented yet")
def test_model_forward_debug():
    from void.nn.model import VoidLM

    cfg = ModelConfig(dim=128, n_layers=4, n_heads=4, n_kv_heads=2, head_dim=32,
                      ffn_dim=384, vocab_size=8192, max_seq_len=256, n_registers=2,
                      attn_pattern=["full"] * 4, sw_size=128)
    m = VoidLM(cfg)
    out = m(torch.randint(0, cfg.vocab_size, (2, 32)))
    assert out["logits"].shape == (2, 32, cfg.vocab_size)
