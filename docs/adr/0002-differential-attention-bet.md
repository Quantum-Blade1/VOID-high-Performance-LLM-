# ADR-0002: Differential Attention as the void_v1 research bet

**Status:** accepted
**Date:** 2026-09-18
**Deciders:** Krish Kumar Sharma
**Consulted:** Ye et al. (2024), "Differential Transformer" (arXiv:2410.05258); `paper/void_v1.tex` §3.3 and §4; ADR-0001.

---

## Context

void_v1's core claim to being a "high-performance" model rather than a stock-Llama clone rests on one novel architectural bet. We must choose which single technique — of the many published in 2024–2025 — to adopt at the sub-1B parameter scale, and formalise this choice as a falsifiable research question so success or failure of the run is unambiguous.

## Decision

Adopt **Differential Attention** (Ye et al. 2024) on every attention layer of void_v1, with the schedule and $\lambda(\cdot)$ parameterisation described in `paper/void_v1.tex` §3.3.

The falsifiable research question is:

> **Does Differential Attention improve code-generation quality at 485M parameters when trained on a code-specialised 100B-token corpus with Stage-3 distillation from Qwen2.5-Coder-7B, versus an otherwise-identical standard GQA baseline?**

## Rationale

**Why one novel bet.** Combining $k$ novel components multiplies debugging surface. Empirically, first-time pretraining projects that stack multiple untested techniques diverge at some step and cannot diagnose which technique failed. We commit to exactly one novelty in v1; MoE, MoD, nGPT, and other candidates are queued for v2+.

**Why Differential Attention.** Among the candidates surveyed (see [Alternatives](#alternatives-considered)):
1. It has a published paper with ablations at 3B parameters showing consistent perplexity improvements and reduced hallucination.
2. It is a localised change (only the attention module); if it fails, the fault is localised.
3. It composes cleanly with every other void_v1 component (GQA, RoPE, sliding-window, QK-norm, MTP).
4. Rollback is trivial: zero the second Q/K projections and freeze $\lambda = 0$.
5. **It is not currently used by any released open model at $\leq 1$B parameters.** This gives void_v1 a genuine "first-of-its-kind" claim.

**Theoretical support.** `paper/void_v1.tex` §4.2 formalises the noise-cancellation argument as Theorem 1 on expected off-diagonal attention mass. The theorem does not prove that diff-attn will improve downstream evals — theory rarely does — but it makes the intuition precise: differential attention reduces the expected attention leaked onto irrelevant tokens by $(1 - \lambda) \cdot \mathbb{E}[\mathcal{N}(\mathcal{N}_c)]$ when common-mode noise assumption holds. Whether this translates to code-generation quality is what we are testing.

## Alternatives considered

| Candidate novel technique | Origin | Reason not chosen for v1 |
|---|---|---|
| Mixture of Depths (MoD) | Google DeepMind 2024 | Router training unstable at small scale; harder to debug |
| nGPT (unit-hypersphere norm) | Nvidia 2024 | Claimed speedups need replication at 100B tokens; restrictive to combine with others |
| Native Sparse Attention (NSA) | DeepSeek 2025 | Complex to implement; benefits mostly at long context (not our v1 target) |
| BitNet b1.58 | Microsoft 2024 | 3-month engineering project to make work at 500M |
| Titans / test-time memory | Google 2025 | Not open; hard to replicate |
| No novel technique | — | Sacrifices the "first-of-its-kind" positioning; makes void_v1 indistinguishable from a well-trained SmolLM clone |
| Two or more novel techniques | — | Multiplied debugging surface; explicitly out of scope per "one bet" discipline |

## Success and failure criteria

The bet is evaluated by an **ablation study** at the end of the main training run:

1. **Primary run (void_v1-diff):** the full architecture with differential attention, trained on the 100B-token curriculum with Stage-3 distillation.
2. **Ablation run (void_v1-baseline):** identical architecture and data, but with differential attention replaced by standard GQA (equivalent to $W_{Q_2} = W_{K_2} = 0$, $\lambda = 0$ frozen). Baseline is trained for **10B tokens only** to make the ablation tractable.

The comparison is made at the 10B-token milestone of both runs — i.e., we run the baseline in full and compare with the primary run's 10B-token checkpoint.

**Bet wins if:**
- HumanEval pass@1 of diff run @ 10B tokens exceeds baseline @ 10B tokens by $\geq 1.5$ points, **AND**
- Perplexity of diff run on held-out validation shard is lower than baseline by $\geq 3\%$.

**Bet loses if:**
- Diff run underperforms baseline on either metric.

**Bet is inconclusive if:**
- Metrics move in opposite directions or improvements are within noise.

## Consequences

**If the bet wins:**
- void_v1 has a real research contribution: *"first sub-1B open code model with Differential Attention, +X on HumanEval, +Y% perplexity."*
- The Stage-2 and Stage-3 training continue with diff-attn.
- Write a technical report on the finding.

**If the bet loses:**
- Enact rollback (§Rollback below).
- Retrain from the 10B-token diff-run checkpoint but with diff-attn disabled, using the same data curriculum.
- Ship as `void-v1-standard`. Publish honest findings: *"Differential Attention did not improve quality at 485M scale for code."* This is itself a valuable negative result.

**If inconclusive:**
- Continue training diff-attn version (default) to full 100B tokens. Note in the model card that the ablation did not resolve the question.

## Rollback plan

Disabling differential attention at any checkpoint is a two-line config change:
```yaml
model:
  attention_type: standard_gqa           # was differential_gqa
```
plus `lambda` schedule ignored. No architectural change to layer count, hidden dim, or any other component. Training can resume from the latest checkpoint in $\leq 1$ hour of setup.

## Open questions

- Whether $\lambda$-schedule should be per-layer learned or globally fixed. Default: per-layer learned via the Ye et al. formula.
- Whether to run the ablation in parallel from step 0 rather than at the 10B milestone. Parallel would give a cleaner comparison but doubles compute. Deferred to a compute-cost decision closer to training start.
- Whether to publish both variants (`void-v1-diff` and `void-v1-baseline`) regardless of which wins, so the community can inspect both. Default: yes, both ship.
