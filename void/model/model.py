"""VoidModel — the top-level decoder-only transformer.

Composes: token embedding, N transformer blocks (see block.py), final RMSNorm,
LM head (tied), and (train-only) MTP heads (see mtp.py).
Register tokens (see registers.py) are prepended at forward time.

Design ref: paper/void_v1.tex §3, docs/architecture.md §3-4."""
