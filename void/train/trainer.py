"""Main training loop.

Handles: forward, backward, gradient accumulation, gradient clipping, optimizer step,
LR scheduling, checkpoint dispatch, logging. Delegates optimizer split, distillation,
and curriculum stage transitions to their respective modules."""
