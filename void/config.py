from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

AttnKind = Literal["gqa", "diff-gqa"]
AttnLayer = Literal["full", "sw"]
Dtype = Literal["float32", "bfloat16", "float16"]


@dataclass
class ModelConfig:
    dim: int = 960
    n_layers: int = 32
    n_heads: int = 15
    n_kv_heads: int = 5
    head_dim: int = 64
    ffn_dim: int = 2560
    vocab_size: int = 49_152
    max_seq_len: int = 2_048

    rope_theta: float = 500_000.0
    norm_eps: float = 1e-5
    qk_norm: bool = True

    attn_kind: AttnKind = "diff-gqa"
    attn_pattern: list[AttnLayer] = field(default_factory=list)
    sw_size: int = 512

    # diff-attn only
    lambda_init_base: float = 0.8
    lambda_init_decay: float = 0.3
    tie_v: bool = True

    n_registers: int = 4
    n_mtp_heads: int = 2
    mtp_loss_weight: float = 0.3

    tie_embeddings: bool = True
    init_std: float = 0.02

    def __post_init__(self) -> None:
        if self.n_heads % self.n_kv_heads:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if self.head_dim * self.n_heads != self.dim:
            raise ValueError("head_dim * n_heads must equal dim")
        if self.attn_pattern and len(self.attn_pattern) != self.n_layers:
            raise ValueError("attn_pattern length must match n_layers")


@dataclass
class OptimConfig:
    hidden_kind: Literal["adamw", "muon"] = "muon"
    hidden_lr: float = 6e-4
    other_lr: float = 3e-4
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.1
    warmup_steps: int = 2_000
    min_lr_frac: float = 0.1
    grad_clip: float = 1.0
    z_loss: float = 1e-4


@dataclass
class TrainConfig:
    seq_len: int = 2_048
    micro_batch: int = 8
    grad_accum: int = 32
    total_tokens: int = 100_000_000_000

    dtype: Dtype = "bfloat16"
    grad_ckpt: bool = True
    flash_attn: bool = True
    liger: bool = True

    optim: OptimConfig = field(default_factory=OptimConfig)

    ckpt_interval: int = 1_000
    ckpt_keep: int = 5
    hub_repo: str | None = None

    log_interval: int = 50
    wandb_project: str | None = None

    seed: int = 42


@dataclass
class Stage:
    name: str
    tokens: int
    mix: str
    teacher: str | None = None
    teacher_beta: float = 0.5
    teacher_temp: float = 2.0


@dataclass
class SFTConfig:
    dataset: str | None = None
    epochs: int = 3
    batch: int = 16
    lr: float = 2e-5


@dataclass
class DPOConfig:
    dataset: str | None = None
    beta: float = 0.1
    epochs: int = 1
    lr: float = 5e-7


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    stages: list[Stage] = field(default_factory=list)
    sft: SFTConfig = field(default_factory=SFTConfig)
    dpo: DPOConfig = field(default_factory=DPOConfig)

    tokenizer_repo: str | None = None
    corpus_repo: str | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        return _from_dict(cls, yaml.safe_load(Path(path).read_text()))


def _from_dict(cls: type, data: Any) -> Any:
    if data is None or not is_dataclass(cls):
        return data
    kwargs: dict[str, Any] = {}
    type_hints = {f.name: f.type for f in fields(cls)}
    for f in fields(cls):
        if f.name not in data:
            continue
        val = data[f.name]
        # Nested dataclass fields.
        if is_dataclass(f.type):
            kwargs[f.name] = _from_dict(f.type, val)
        elif getattr(f.type, "__origin__", None) is list and val:
            (item_t,) = f.type.__args__
            kwargs[f.name] = [_from_dict(item_t, v) for v in val]
        else:
            kwargs[f.name] = val
    return cls(**kwargs)
