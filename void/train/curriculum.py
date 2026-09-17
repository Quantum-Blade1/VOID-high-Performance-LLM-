"""Three-stage curriculum orchestrator.

Reads VoidConfig.curriculum; produces a stage iterator that yields data-mix configs
and LR schedule anchors per stage. Transitions data mix and (in stage 3) enables the
distillation loss.

Design ref: paper/void_v1.tex §5.2, docs/architecture.md §8."""
