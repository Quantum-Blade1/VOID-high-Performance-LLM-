"""DifferentialGQAAttention — the novel research bet.

Implements two independently-projected Q and K branches with shared V and per-layer
learnable lambda(t). Falls back to standard GQA when attention_type == 'standard_gqa'.
QK-norm is applied to both branches. RoPE is applied post-projection, post-norm.
Supports full attention or sliding-window (window = VoidConfig.model.sliding_window_size).

Design refs: paper/void_v1.tex §3.3, §4.2 (Theorem 1); ADR-0002."""
