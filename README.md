# TinyLM Lab

A reproducible experimental framework for studying how miniature Transformer
language models learn: synthetic data generation, a configurable tiny
Transformer, and a training/evaluation pipeline with logged, reproducible
metrics.

**Research question:** How do model capacity, dataset size and task difficulty
affect memorisation, generalisation and overfitting in miniature Transformers?

All three factors in the RQ run as mainline one-factor-at-a-time series,
with a grokking hunt at the modular-arithmetic extreme of the difficulty series.

Honours project (CS41001), University of Dundee. Supervisor: Dr John Lawrence.

## Status

**Prototype complete** - tiny Transformer on PyTorch primitives (≈0.4M params),
and a minimal end-to-end pipeline (one synthetic
task → seeded training loop → logged train/val metrics → one command → one
plot). Build spec: [docs/planning/prototype-spec.md](docs/planning/prototype-spec.md).
`pytest` is green (17 tests), including a CPU bit-exact same-seed reproducibility test.

**Reporting:** the bit-exact same-seed guarantee is asserted on CPU, which makes CPU the
canonical backend (MPS is faster but nondeterministic, so opt-in only). Held-out patterns
share no pattern-to-pattern mapping with training patterns, so flat or rising validation
curves are expected and reported as-is.

## Usage

```bash
uv run python -m tinylm.train --config configs/copy.yaml
# → logs/<run>/metrics.jsonl + learning_curve.png

uv run pytest
```

## Documentation

- [Project log](LOG.md) - meeting and decision record
- [Whole-project plan](docs/planning/plan.md) - phases, rubric mapping, LSEP, Gantt
- [Prototype spec](docs/planning/prototype-spec.md) - exactly what the prototype build contains
- [Reading list](docs/reading-list-seeds.md) - branch map + parked sources (references live in Zotero)

## Licence

[MIT](LICENSE)
