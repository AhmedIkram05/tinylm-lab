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

**Prototype in development** - a hand-written tiny Transformer on PyTorch
primitives, plus a minimal end-to-end pipeline (one synthetic task → seeded
training loop → logged train/val metrics → one command → one plot). Build
spec: [docs/planning/prototype-spec.md](docs/planning/prototype-spec.md).

**Reporting:** training runs on MPS (Apple Silicon); the reproducibility
guarantee is pinned on CPU - one test asserts bit-exact same-seed metrics
there, while MPS may vary across drivers. Held-out patterns share no
pattern-to-pattern mapping with training patterns, so flat or rising
validation curves are expected and reported as-is.

<!--
## Usage

```bash
uv run python -m tinylm.train --config configs/copy.yaml
# → logs/<run>/metrics.jsonl + learning_curve.png
```
-->

## Documentation

- [Project log](docs/LOG.md) - meeting and decision record
- [Whole-project plan](docs/planning/plan.md) - phases, rubric mapping, LSEP, Gantt
- [Prototype spec](docs/planning/prototype-spec.md) - exactly what the prototype build contains
- [Decision records](docs/adr/) - the why behind the design
- [Reading list](docs/reading-list-seeds.md) - branch map + parked sources (references live in Zotero)

## Licence

[MIT](LICENSE)
