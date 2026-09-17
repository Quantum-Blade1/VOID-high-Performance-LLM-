from __future__ import annotations

import pytest

from void.config import OptimConfig
from void.train.optim import cosine_schedule


def test_cosine_warmup_to_peak():
    cfg = OptimConfig(warmup_steps=100, min_lr_frac=0.1)
    assert cosine_schedule(0, 1.0, cfg, total_steps=1_000) == pytest.approx(0.0)
    assert cosine_schedule(100, 1.0, cfg, total_steps=1_000) == pytest.approx(1.0)


def test_cosine_end_is_min():
    cfg = OptimConfig(warmup_steps=100, min_lr_frac=0.1)
    assert cosine_schedule(1_000, 1.0, cfg, total_steps=1_000) == pytest.approx(0.1)


@pytest.mark.skip(reason="train loop not implemented")
def test_debug_run_decreases_loss():
    ...
