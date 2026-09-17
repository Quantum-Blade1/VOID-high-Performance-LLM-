"""Register tokens — 4 learnable prepended tokens.

These positions are masked in the loss and serve as dedicated attention sinks,
reducing attention entropy on the real content tokens.

Design ref: Darcet et al. 2024 (Vision Transformers Need Registers)."""
