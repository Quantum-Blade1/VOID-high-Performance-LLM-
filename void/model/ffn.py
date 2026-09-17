"""SwiGLU feed-forward network.

Forward: y = (SiLU(x @ W_gate) * (x @ W_up)) @ W_down.
No bias in any linear layer.

Design ref: Shazeer 2020 (GLU Variants Improve Transformer)."""
