"""Cosine LR schedule with linear warmup.

Computes lr(step) = min_lr + 0.5 * (peak_lr - min_lr) * (1 + cos(pi * progress))
where progress = (step - warmup_steps) / (total_steps - warmup_steps).
Supports resume-from-step via configuration."""
