"""Checkpoint save / load / push to HF Hub.

Persists: model weights (safetensors), optimizer state (both AdamW and Muon),
LR scheduler state, RNG states (python/numpy/torch/cuda), data iterator position
(shard + offset), step number, and elapsed wall-clock.
Atomic writes; background upload to HF Hub. Keeps last N and all milestone checkpoints."""
