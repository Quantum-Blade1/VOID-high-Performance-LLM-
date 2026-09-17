"""VoidConfig — dataclass container for the full model + training configuration.

Loading from YAML::

    from void.config import VoidConfig
    cfg = VoidConfig.from_yaml("configs/void_v1_500m.yaml")

The dataclass mirrors the schema of ``configs/void_v1_500m.yaml``. Every field
here has a corresponding entry in that file; if you add a field, add it there
too and vice-versa. The debug config ``configs/void_v1_debug_10m.yaml`` uses
the same schema at a tiny scale.

Design references:
    - paper/void_v1.tex, §3 (architecture) and §5 (training)
    - docs/architecture.md
    - docs/adr/0001-architecture.md
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------


@dataclass
class DifferentialConfig:
    """Parameters specific to Differential Attention (ADR-0002)."""

    lambda_init_decay: float = 0.3
    lambda_init_base: float = 0.8
    tie_v_across_branches: bool = True


@dataclass
class MTPConfig:
    """Multi-Token Prediction (train-only) heads."""

    enabled: bool = True
    num_heads: int = 2
    loss_weights: list[float] = field(default_factory=lambda: [0.3, 0.3])
    tie_output_head: bool = True
    per_head_transformer_layers: int = 1


@dataclass
class InitConfig:
    std: float = 0.02
    scale_residual_projs_by: Literal["1_over_sqrt_2L", "none"] = "1_over_sqrt_2L"


@dataclass
class ModelConfig:
    """All shape / architecture hyperparameters. Corresponds to configs YAML `model:` block."""

    # Shape
    hidden_dim: int = 960
    num_layers: int = 32
    num_query_heads: int = 15
    num_kv_heads: int = 5
    head_dim: int = 64
    ffn_intermediate_dim: int = 2560
    vocab_size: int = 49152
    max_seq_len: int = 2048
    tie_word_embeddings: bool = True

    # Positional
    rope_theta: float = 500_000.0
    rope_scaling: Any = None

    # Norm
    rms_norm_eps: float = 1.0e-5
    qk_norm: bool = True

    # Attention
    attention_type: Literal["standard_gqa", "differential_gqa"] = "differential_gqa"
    differential: DifferentialConfig = field(default_factory=DifferentialConfig)

    # Mixed attention pattern (per-layer 'full' | 'sliding_window'); len == num_layers.
    attention_pattern: list[str] = field(default_factory=list)
    sliding_window_size: int = 512

    # Register tokens
    num_register_tokens: int = 4

    # Multi-Token Prediction
    mtp: MTPConfig = field(default_factory=MTPConfig)

    # FFN
    ffn_activation: Literal["swiglu"] = "swiglu"

    # Init
    init: InitConfig = field(default_factory=InitConfig)

    # Precision
    dtype: Literal["float32", "bfloat16", "float16"] = "bfloat16"
    master_weights_dtype: Literal["float32", "bfloat16"] = "float32"

    def __post_init__(self) -> None:
        # Basic sanity checks; full validation is delegated to model construction.
        if self.num_query_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_query_heads ({self.num_query_heads}) must be divisible by "
                f"num_kv_heads ({self.num_kv_heads})"
            )
        if self.head_dim * self.num_query_heads != self.hidden_dim:
            raise ValueError(
                f"head_dim * num_query_heads ({self.head_dim} * {self.num_query_heads}) "
                f"must equal hidden_dim ({self.hidden_dim})"
            )
        if self.attention_pattern and len(self.attention_pattern) != self.num_layers:
            raise ValueError(
                f"attention_pattern length ({len(self.attention_pattern)}) must equal "
                f"num_layers ({self.num_layers})"
            )


@dataclass
class OptimizerGroupConfig:
    name: Literal["adamw", "muon"] = "adamw"
    lr: float = 3.0e-4
    momentum: float = 0.95              # Muon-only
    newton_schulz_iters: int = 5        # Muon-only
    betas: list[float] = field(default_factory=lambda: [0.9, 0.95])  # AdamW-only
    eps: float = 1.0e-8
    weight_decay: float = 0.1


@dataclass
class OptimizerConfig:
    hidden: OptimizerGroupConfig = field(default_factory=OptimizerGroupConfig)
    embed_head_norm: OptimizerGroupConfig = field(default_factory=OptimizerGroupConfig)


@dataclass
class SchedulerConfig:
    warmup_steps: int = 2000
    schedule: Literal["cosine", "linear", "constant"] = "cosine"
    min_lr_fraction: float = 0.1


@dataclass
class CheckpointConfig:
    interval_steps: int = 1000
    keep_last_n: int = 5
    milestones_step: list[int] = field(default_factory=list)
    push_to_hub: bool = False
    hub_repo: str | None = None
    hub_private_until_release: bool = True


@dataclass
class WandbConfig:
    project: str | None = None
    entity: str | None = None


@dataclass
class LoggingConfig:
    log_interval_steps: int = 50
    wandb: WandbConfig = field(default_factory=WandbConfig)


@dataclass
class TrainingConfig:
    seq_len: int = 2048
    micro_batch_size: int = 8
    grad_accum_steps: int = 32
    total_tokens: int = 100_000_000_000

    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)

    grad_clip_norm: float = 1.0
    z_loss_coeff: float = 1.0e-4
    seed: int = 42

    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    use_flash_attention: bool = True
    use_liger_kernel: bool = True
    gradient_checkpointing: bool = True
    gradient_checkpoint_every_n_layers: int = 4


@dataclass
class DistillationConfig:
    enabled: bool = False
    teacher_model: str | None = None
    temperature: float = 2.0
    beta: float = 0.5
    cache_top_k: int = 100
    teacher_logits_repo: str | None = None


@dataclass
class CurriculumStage:
    name: str
    target_tokens: int
    data_mix_config: str
    lr_schedule_phase: str
    distillation: DistillationConfig = field(default_factory=DistillationConfig)


@dataclass
class CurriculumConfig:
    stages: list[CurriculumStage] = field(default_factory=list)


@dataclass
class DataConfig:
    tokenizer_repo: str | None = None
    corpus_repo: str | None = None
    streaming: bool = True
    num_workers: int = 8
    prefetch_factor: int = 2
    pin_memory: bool = True
    persistent_workers: bool = True


@dataclass
class FailFastConfig:
    humaneval_at_step_20000_min: int = 8


@dataclass
class EvaluationConfig:
    benchmarks: list[str] = field(default_factory=list)
    eval_at_steps: list[int] = field(default_factory=list)
    eval_batch_size: int = 32
    fail_fast: FailFastConfig = field(default_factory=FailFastConfig)


@dataclass
class SFTConfig:
    enabled: bool = True
    dataset: str | None = None
    epochs: int = 3
    batch_size: int = 16
    lr: float = 2.0e-5
    format: Literal["chatml"] = "chatml"


@dataclass
class DPOConfig:
    enabled: bool = True
    dataset: str | None = None
    beta: float = 0.1
    epochs: int = 1
    lr: float = 5.0e-7


@dataclass
class PostTrainingConfig:
    sft: SFTConfig = field(default_factory=SFTConfig)
    dpo: DPOConfig = field(default_factory=DPOConfig)


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------


@dataclass
class VoidConfig:
    """Top-level configuration container for void_v1."""

    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    data: DataConfig = field(default_factory=DataConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    post_training: PostTrainingConfig = field(default_factory=PostTrainingConfig)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> "VoidConfig":
        """Load a VoidConfig from a YAML file.

        Implementation note: this is a straightforward nested-dict → dataclass
        mapping; we deliberately avoid pulling in `omegaconf` or `hydra` to
        keep dependencies minimal. If the schema grows more complex, revisit.
        """
        raise NotImplementedError("Implement in ADR-0003 or during model scaffolding.")

    def to_dict(self) -> dict[str, Any]:
        """Serialise back to a plain dict (for logging / wandb / model card)."""
        raise NotImplementedError
