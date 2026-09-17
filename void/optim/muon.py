"""Muon optimizer.

Applies Newton-Schulz orthogonalization to gradient updates on 2D+ matrices.
Only used on hidden weight matrices; embedding, LM head, RMSNorm weights, and
lambda scalars are routed to AdamW instead (see param_groups.py).

Design ref: Keller Jordan 2024 (Muon: An Optimizer for Hidden Layers)."""
