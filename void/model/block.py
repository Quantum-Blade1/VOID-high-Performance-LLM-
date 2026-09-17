"""TransformerBlock — pre-norm block: RMSNorm → attention → residual, RMSNorm → SwiGLU → residual.

Attention type is switched per-layer (full vs sliding-window) via VoidConfig.model.attention_pattern.
The attention module itself is Differential GQA (see attention.py) or standard GQA depending on config.

Design ref: docs/architecture.md §4."""
