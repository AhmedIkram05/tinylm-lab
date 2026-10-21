# TinyLM Lab

A reproducible experimental framework for studying how miniature Transformer language models learn: synthetic data generation, a configurable tiny Transformer, and a training evaluation pipeline with logged, reproducible metrics.

**Research question:** How do model capacity, dataset size and task difficulty affect memorisation and generalisation in miniature Transformers trained on synthetic sequence-copying and modular-arithmetic tasks?

**Sub-question:** Under what conditions does delayed generalisation (grokking) emerge on the modular-arithmetic task?

All three factors in the RQ run as mainline one-factor-at-a-time series, with the grokking sub-question riding the modular-arithmetic thread; sequence copying and modular arithmetic are two task families with their own difficulty controls, not one difficulty scale. [docs/planning/plan.md](docs/planning/plan.md).

Honours project (CS41001), University of Dundee. Supervisor: Dr John Lawrence.

## Status

**Prototype complete** - tiny Transformer on PyTorch primitives (≈0.4M params), and a minimal end-to-end pipeline (two synthetic tasks → seeded training loop → logged train/val metrics → one command → one plot). Build spec: [docs/planning/prototype-spec.md](docs/planning/prototype-spec.md). `pytest` is green (39 tests), including a CPU bit-exact same-seed reproducibility test.

**Tasks:** sequence copying (`configs/copy.yaml`) and modular arithmetic (`configs/modadd.yaml` + `configs/modadd_wd0.yaml`) — sequences `[a, PLUS, b, EQ, (a+b) % modulus]` over 11 value tokens, `train_frac` of the 121 ordered pairs.

**Reporting:** the bit-exact same-seed guarantee is asserted on CPU, which makes CPU the canonical backend (MPS is faster but nondeterministic, so opt-in only). Held-out patterns share no pattern-to-pattern mapping with training patterns, so flat or rising validation curves are expected and reported as-is.

## Usage

```bash
uv run python -m tinylm.train --config configs/copy.yaml
# → logs/<run>/metrics.jsonl + learning_curve.png + logs/<run>/eval_positions.jsonl
#   (per-position arrays have seq_len - 1 entries: 31 for copy, 4 for modadd)

uv run python -m tinylm.train --config configs/modadd.yaml

uv run pytest
```

Sweeps: `scripts/sweep.py --sweep <descriptor> --out <dir>` runs one subprocess per
(config, seed) and writes a `registry.jsonl` that includes failed runs;
`plot_gap.py` / `plot_trajectory.py` draw the gap-vs-factor and val-acc-vs-step
figures, and `registry_to_csv.py` writes the CSV evidence.

## Documentation

- [Project log](LOG.md) - meeting and decision record
- [Whole-project plan](docs/planning/plan.md) - phases, rubric mapping, LSEP, Gantt
- [Prototype spec](docs/planning/prototype-spec.md) - exactly what the prototype build contains

## Licence

[MIT](LICENSE)
