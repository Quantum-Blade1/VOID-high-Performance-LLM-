"""Repository-aware sequence packing for Stack v2.

Within a repo, concatenate related files (topologically sorted by import graph) into
the same 2048-token sequence. Cross-file context boosts HumanEval-style evals by
5-10 points vs random-shuffled file training (DeepSeek-Coder report)."""
