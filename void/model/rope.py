"""Rotary Position Embedding.

Precomputes cos/sin tables for positions [0, max_seq_len). Applied to Q and K
after projection and QK-norm, before the attention dot product.
theta is configurable (default 500,000 for RoPE extension headroom).

Design ref: Su et al. 2024 (RoFormer)."""
