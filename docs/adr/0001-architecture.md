# ADR-0001: void_v1 model architecture

**Status:** accepted
**Date:** 2026-09-18
**Deciders:** Krish Kumar Sharma
**Consulted:** `paper/void_v1.tex`, `docs/architecture.md`, and the cited references therein.

---

## Context

We are building **void_v1**, a ~485M-parameter decoder-only language model targeted at code generation and command-line reasoning. We must choose a concrete architecture before writing any code. The design must:

1. Be trainable from scratch on ~100B tokens using Kaggle-tier free compute or a modest rented-GPU budget (< $1000).
2. Aim for competitive quality with published sub-1B code models (Qwen2.5-Coder-0.5B, DeepSeek-Coder-1.3B) rather than a stock-Llama baseline.
3. Be debuggable and instrumentable at solo-builder scale — no black-box or opaque components.
4. Leave graceful-degradation paths so any single novel component can be disabled without abandoning the project.

## Decision

Adopt the architecture specified in `paper/void_v1.tex` §3 and `docs/architecture.md`:

- **480–500M parameters, decoder-only transformer** with tied input/output embeddings.
- **Deep-and-narrow shape:** hidden dim 960, 32 layers.
- **GQA 3:1** (15 query heads, 5 KV heads, head dim 64).
- **Differential Attention** on every layer (two Q + two K projections, shared V, learnable per-layer $\lambda$).
- **QK-norm** on both diff-attention branches.
- **Mixed attention pattern:** layers 1–4 and 25–32 use full attention; layers 5–24 use `[SW-512, SW-512, SW-512, full]` repeated 5 times.
- **RoPE with $\theta = 500{,}000$** to allow later context extension.
- **SwiGLU FFN** with inner dim 2560.
- **4 register tokens** prepended to every training sequence.
- **Multi-Token Prediction:** 2 auxiliary heads (t+2, t+3), train-only, loss weights 0.3 each.
- **RMSNorm** everywhere (pre-norm placement on both attention and FFN sub-layers).
- **Muon** optimizer for hidden matrices, **AdamW** for embedding / LM head / norms / $\lambda$-scalars.
- **BPE tokenizer** with vocab size 49,152, FIM + ChatML special tokens.

## Rationale

- **Depth over width** — MobileLLM ablations show that at $\leq 1$B params, depth dominates width per parameter. 32 layers on 960 hidden matches SmolLM2-360M's shape; we scale up total params by widening FFN and using diff-attn's extra projections.
- **GQA** — Llama-3-standard; saves ~8% of params and 5/3× the KV cache without measurable quality loss.
- **Differential Attention** — the single novel bet at this scale; see [ADR-0002](0002-differential-attention-bet.md).
- **QK-norm** — Chameleon / ViT-22B evidence; cheap late-training stability insurance and required for stable differential attention.
- **Mixed attention pattern** — Mistral / Gemma-2 pattern; ~40% attention FLOP savings with the 12 full-attention layers still sufficient for long-range integration.
- **MTP heads** — DeepSeek-V3 shows 1.5–2× data efficiency during pretraining; free acceleration at trivial parameter cost.
- **Register tokens** — Darcet et al. shows register sinks reduce attention entropy on real tokens; particularly synergistic with differential attention where uncontrolled noise is what we are trying to cancel.
- **Muon** — 1.5–2× sample efficiency claimed in nanoGPT speedrun; the primary lever for closing the 55× token-budget gap between us and Qwen2.5-Coder-0.5B.

## Alternatives considered

| Option | Pros | Cons | Reason not chosen |
|---|---|---|---|
| Stock Llama-3-500M | Boring but proven | Zero differentiation | Doesn't earn the "high-performance" label; identical to what everyone else ships |
| MoE (8 experts, top-2) | Higher active param quality | Training instability at solo scale; deployment friction | Deferred to void_v2; combined novelty budget too large for v1 |
| Mamba-Transformer hybrid | Long-context potential | Small-model ablations underperform pure transformer | Research literature is not yet favorable at 500M |
| Full novel architecture | Maximum research reward | Every debugging surface novel; graveyard | Explicitly out of scope for a first pretraining project |
| MobileLLM shape verbatim (350M) | Proven | Doesn't satisfy the ~500M ask | Kept the shape idea, scaled to 480M via FFN + diff-attn |

## Consequences

**Positive**
- One controlled novelty (differential attention) means training failures are diagnosable.
- Every component has a paper citation → the design is defensible in a model card.
- Rollback paths mean no scenario produces "spent 3 months, shipped nothing."

**Negative**
- Differential attention doubles Q/K projections → ~8% param overhead vs. standard GQA, and roughly +15–20% attention compute.
- Muon is less battle-tested than AdamW; potential instability requires a fallback plan.
- The mixed attention pattern complicates the model export path (some inference backends prefer uniform attention).

**Reversibility.** Each component can be individually reverted at the cost of retraining from a checkpoint. See [Rollback plan](#rollback-plan). No architectural choice is truly irreversible before the first release.

## Rollback plan

| Component | Rollback | Cost |
|---|---|---|
| Differential Attention | Set $W_{Q_2} = W_{K_2} = 0$, freeze $\lambda = 0$ → standard GQA | Config-only; re-train from any checkpoint |
| Multi-Token Prediction | Set aux loss weights to 0 | Config-only; heads become dead params but don't affect inference |
| Register tokens | Drop from tokenized sequences; adjust position IDs | Requires re-tokenization pass; ~1 day |
| Muon optimizer | Swap to pure AdamW at any checkpoint | Config-only; may briefly perturb loss curve |
| Mixed attention | Convert all SW layers to full | Config-only; +40% attention FLOPs |
| QK-norm | Set QK-norm weight to identity, freeze | Config-only |

## Open questions

- Exact $\lambda_\text{init}$ schedule vs. per-layer learned init. We follow the Ye et al. schedule but may ablate.
- Whether to allow the two Q/K projections in differential attention to share parameters partially (e.g., LoRA-style low-rank second branch) to reduce the 8% param overhead.
- Whether MTP heads should share weights beyond the LM head tie (e.g., share their small transformer blocks).

These will be resolved either by (a) preliminary ablations on the debug config, or (b) fixed at defaults and revisited only if training misses milestones.
