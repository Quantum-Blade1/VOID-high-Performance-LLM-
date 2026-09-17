from __future__ import annotations

from pathlib import Path

import pytest

from void.config import Config, ModelConfig

CONFIGS = Path(__file__).parent.parent / "configs"


def test_defaults_ok():
    cfg = Config()
    assert cfg.model.dim == 960
    assert cfg.model.n_heads == 15


@pytest.mark.parametrize("path", ["500m.yaml", "debug.yaml"])
def test_yaml_roundtrip(path):
    cfg = Config.from_yaml(CONFIGS / path)
    assert cfg.model.n_heads % cfg.model.n_kv_heads == 0


def test_invalid_head_layout():
    with pytest.raises(ValueError):
        ModelConfig(n_heads=16, n_kv_heads=5)
