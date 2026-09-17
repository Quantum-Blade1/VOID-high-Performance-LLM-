# void

A 485M-parameter decoder-only LM for code and command-line reasoning. Design-phase; no training runs yet.

The one deliberate bet: differential attention (Ye et al., 2024) at sub-1B scale. Everything else is drawn from published models with reproducible ablations.

## Layout

```
void/           model, data, training, eval, post, export
configs/        500m.yaml, debug.yaml
scripts/        train, data, eval, export, bench
tests/
docs/           architecture.md, adr/
paper/          void_v1.tex + PDF
```

## Docs

- [`paper/void_v1.pdf`](paper/void_v1.pdf) — design paper with proofs and derivations
- [`docs/architecture.md`](docs/architecture.md) — architecture reference
- [`docs/adr/`](docs/adr) — decision records

## Build the paper

```bash
cd paper && tectonic void_v1.tex
```

## License

Apache-2.0.
