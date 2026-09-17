"""Multi-Token Prediction heads (training only).

Two auxiliary heads predict tokens at t+2 and t+3 from the final hidden state.
Each head is a 1-layer transformer block + linear projection to the shared LM head.
Discarded at inference.

Design ref: Gloeckle et al. 2024, DeepSeek-V3 report; docs/architecture.md §6."""
