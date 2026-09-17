"""Token embedding module with tied LM head weights.

The output logits are computed as h @ embedding.weight.T (weight tying).
Init: N(0, 0.02)."""
