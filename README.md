# VOID — High-Performance Sub-1B Code LLM

**void_v1** is a 485M-parameter decoder-only language model designed to be trained from scratch on ~100B tokens of a curated code + general corpus. It targets competitive performance with published sub-1B code models (Qwen2.5-Coder-0.5B, DeepSeek-Coder-1.3B) while planting one research flag: **Differential Attention at the sub-1B scale**, a mechanism proposed by Microsoft Research in late 2024 that has not yet appeared in any released model of this size.

## Status

**Design phase — pre-training.** No training runs have executed yet. This repository currently contains the design specification and theoretical analysis.

## Repository layout

```
.
├── paper/
│   ├── void_v1.tex     LaTeX research design paper with proofs
│   └── void_v1.pdf     Compiled PDF
└── docs/
    └── architecture.md Full architecture documentation with Mermaid diagrams
```

## Documents

- **[`paper/void_v1.pdf`](paper/void_v1.pdf)** — the research design paper. Includes formal theorems, closed-form parameter and FLOP derivations, and theoretical comparison with five reference models.
- **[`docs/architecture.md`](docs/architecture.md)** — human-readable architecture reference with system-level and block-level Mermaid diagrams.

## The one novel bet

> Does Differential Attention improve code-generation quality at 485M parameters when trained on a code-specialized 100B-token corpus with Stage-3 distillation from Qwen2.5-Coder-7B, versus an otherwise-identical standard GQA baseline?

Every other component of void_v1 is drawn from published models with reproducible ablations. Every novel or non-standard component has a documented rollback path in the risk register (see `docs/architecture.md` §13).

## Architecture summary

| Property | Value |
|---|---|
| Parameters | ~485M |
| Layers | 32 (deep-and-narrow) |
| Hidden dim | 960 |
| Heads | 15 Q / 5 KV (GQA 3:1) |
| Attention | **Differential GQA** (2 Q + 2 K projections) |
| Attention pattern | Mixed sliding-window 512 + full |
| FFN | SwiGLU, inner dim 2560 |
| Vocab | 49,152 BPE + FIM + ChatML specials |
| Context | 2048 (extensible to 32k via RoPE θ=500k) |
| Extras | Register tokens (4), MTP heads (2), QK-norm |
| Optimizer | Muon (hidden) + AdamW (embed/head/norms) |
| Training | 100B tokens, three-stage curriculum |

## Compiling the paper

```bash
cd paper && tectonic void_v1.tex
```

`tectonic` auto-fetches any required TeX packages. Alternatives: `pdflatex void_v1.tex` twice (for ToC and references to resolve).

## License

TBD — will be finalized before the first model release. The intent is a permissive open license (Apache 2.0 or MIT).
