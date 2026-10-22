# Prototype hardening spec - "the pipeline is trusted"

**Status:** Completed (22 Oct 2026)

**Goal:** In the hardening window (9-22 Oct), make the prototype trustworthy enough that a ~330-run experiment grid can be launched and believed: a machine-triageable CLI, data and metric layers that carry both task families and a size-independent comparison, an in-loop memorisation measurement, a sweep runner with committed evidence, and loud failures for unattended runs. No new *code* is written during the sweeps - the research grid is authored afterwards as pure YAML over the fields hardening lands.

## Constraints

- Builds on top of the prototype-spec.
- `uv run --frozen` everywhere: `uv.lock` carries the CPU reproducibility claim, `pyproject.toml` may float.
- Laptop-only, CPU canonical, MPS opt-in parity only; every variant stays under the ≤2M parameter ceiling.
- The tooling is on the experiment phase's zero-code critical path and is never descoped; only accessories (the parity tolerance, conditional fallbacks, the edge tests, the long smokes) would have been cut if the window had forced it.

## What hardening must not change (the repro contract)

- The copy loop is pinned: `AdamW(lr=cfg.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=cfg.weight_decay)` with copy configs at `weight_decay: 0.0`; constant LR, no schedule, no gradient clipping; LM-shift cross-entropy over positions 1..T−1 with accuracy over the copyable window `period..T-1`. Mod-arith is a declared, scoped exception to weight decay only.
- `logs/<run>/metrics.jsonl` keeps its exact 10-column schema (`step, train_loss, train_acc, val_loss, val_acc, lr, elapsed, seed, config_hash, torch_version`). Every new metric goes to a sidecar file, never to a new column.
- The RNG protocol is fixed: `set_seeds(cfg.seed)` → `generate_splits` (numpy `default_rng(cfg.seed)`) → `TinyLM(cfg)`, with `make_train_generator` an independent `torch.Generator`. Nothing added inside the training loop may consume the global stream, so the trajectory stays value-identical to the frozen merge on every column except `elapsed` and `config_hash`.
- `config_hash` covers `asdict(cfg)`, so adding any `Config` field rotates it; it is therefore compared against its pinned value separately, never folded into the trajectory equality.
- `configs/copy.yaml` stays byte-identical - it is the demo anchor and the canonical-CPU config - so every variant is a new file.
- The programme seed set is {0..9}, pre-registered and never expanded; statistics are mean ± sample std (`statistics.stdev`, ddof=1) with `spread` = max − min.
- `sweeps/smoke.yaml` is the sole carve-out from the 10-seed rule: a runner acceptance fixture (2 ok runs + 1 deliberate failure), not a programme claim.
- `.gitignore` gains one line, `!docs/evidence/**/*.png`, so the retained evidence actually commits.
- The frozen prototype spec is not edited; its Status stays "Completed".

## Spec

- **CLI** (`tinylm/train.py`): an additive `--seed INT` beside `--steps`/`--device`, applied before `validate()` and reflected in `config_hash`. The same `main()` edit wraps config loading, the overrides, validation and a `get_device` probe in one handler: a bad path, malformed YAML, failed validation and an unknown device all exit 2 with `config error: …` on stderr and create no `logs/<run>` directory. `run_training` stays outside that wrap, so genuine runtime errors keep their natural exit code - 40-80 subprocess runs need machine-triageable codes.
- **Data** (`tinylm/data.py`): one `generate_splits` dispatching on `Config.task`. The copy path gains `common_val`: a size-independent val pool of the first 512 drawn patterns, with train starting at the same index for every size, so the train sets are strictly nested and `val` is byte-identical across sizes. The probe becomes `train[:min(128, n_train)]`, which makes an n=64 pilot legal, and under `common_val` it is drawn from that same selected window so it is identical across every size ≥ 128. Vocab coverage raises a `ValueError` rather than asserting, so the message survives `-O` and reaches the sweep runner's `stderr_tail`. The mod-arith path builds ordered `(a, b)` pair splits and emits `[a, PLUS, b, EQ, (a+b) % modulus]` with `PLUS = modulus` and `EQ = modulus + 1`.
- **Config** (`tinylm/utils.py`): gains `common_val`, plus `task` / `modulus` / `train_frac`. `train_frac` derives `n_train = int(train_frac × modulus²)` and `n_val = modulus² − n_train` in `__post_init__`, before validation, so the sweep runner and the child process derive identically and therefore hash identically. `validate()` branches onto the task: the `period`/`seq_len` rules are copy-only; mod-arith requires `seq_len == 5`, `vocab_size == modulus + 2` and `period == seq_len − 1`, which pins the answer position and so makes `val_acc` answer accuracy by construction; and `seed` is checked into `[0, 2**32)` so a bad seed is a config error rather than a crash inside `set_seeds`.
- **Instrumentation** (`tinylm/metrics.py` + `train.py`): `metrics.py` holds the loss and every metric - the scalar pair, per-position vectors, and per-example span metrics. Each eval writes two sidecars beside `metrics.jsonl`, at the same cadence and from the same single forward: `eval_positions.jsonl` (per-position loss and accuracy, length T−1, never assumed to be 31) and `eval_examples.jsonl` (a per-example exact-reproduction indicator plus span CE, for the fixed train probe *and* the full val set). Reproduction is conjunctive exact argmax match over the task's own span - the copyable window for copy, the answer token for mod-arith, one expression because `period == seq_len − 1` there - so it is threshold-free and no ε is pre-registered. This is the memorisation measurement, and the only record of *when* memorisation happened. Each run also writes a final `model.pt` checkpoint as re-analysis insurance (~1.6 MB at the demo scale; resume stays parked), and `forward(return_attention=True)` returns every layer's attention for the conditional mechanism snapshot.
- **Configs**: `configs/copy.yaml` stays frozen; the dataset-size series adds `copy_n064/n256/n512/n1024.yaml` (all `common_val: true`); mod-arith adds `modadd.yaml` and the `modadd_wd0.yaml` weight-decay ablation. No seed-variant YAMLs anywhere - every seed comes from `--seed`.
- **Sweep tooling** (`scripts/`): `sweep.py` runs one subprocess per `<config, seed>`, sequentially, with a per-sweep `timeout_s` so a hung run cannot wedge an axis, namespaced run-dirs, a never-overwrite guard, a refusal to re-run an existing registry, and an optional `--filter`. Its `registry.jsonl` records a row for every run *including failures*, carrying `git_commit`, backend, wall time, torch version, and the run's `config_hash` plus a `hash_match` check against the hash inside the run's own log. `plot_gap.py` draws gap-vs-factor with sample-std error bars when seeds exist and points when only one does; `plot_trajectory.py` draws per-run thin lines under a mean ± sample-std band with a 0.9 reference line on `val_acc`, because grokking is a time phenomenon and a final-row gap cannot show it; `registry_to_csv.py` writes the per-run, per-factor and per-step CSVs. All three share the readers in `tinylm/plotting.py` and the colours in `tinylm/palette.py`, and none imports a sibling script.
- **Tests** (`tests/`): The prototype's files cover the model, data, training loop, CPU reproduction and metrics; hardening adds `test_modadd.py` (task determinism, ordered-pair disjointness, answer correctness, sidecar agreement, an overfit smoke, and a golden split hash that pins the copy path), `test_metrics.py` (span semantics plus the subprocess sidecar smoke), `test_edge.py` (odd-but-legal shapes, a parametrized bad-value matrix, and the CLI exit-2 behaviour checked procedurally) and `test_tooling.py` (one test per tooling contract).

## Non-goals (do not build, even if tempting mid-step)

1. Checkpointing as *resume* - a final-state checkpoint is kept as insurance only; whole-axis re-run remains the recovery strategy, and mid-run checkpointing would need another `Config` field.
2. W&B, dashboards, any UI.
3. Larger/fancier models, learned PE, untied embeddings, dropout tuning, copy-WD tuning. The capacity *series* is a different thing: it is the RQ's capacity axis and belongs to the experiment phase, within the ceiling.
4. A multi-task registry, Hydra-style config composition (including per-entry hyperparameter overrides - the grid deliberately uses one config file per point), and parallel sweep execution.
5. Publication-grade plots beyond the two sweep figures; per-position bands and fancy styling stay with the experiment phase.
6. Chasing an absent n=64 gap below n=64, or debugging a non-learning mod-arith pilot past one bounded retry - both routes terminate in an honest finding.
7. An attention *series* and induction scores: attention is not an RQ factor, so no sweep can discriminate on it. All-layer output exists only for a single conditional snapshot.
8. The research grid itself (capacity/difficulty/grok configs and sweep descriptors): pure YAML over fields that already exist, so it is experiment design rather than hardening work.

## Timeline

- 9-22 Oct: the hardening window, cut from the prototype merge; steps 1-7 below.
- Every step ends with `uv run --frozen pytest` green plus a 50-step CPU floor run, so the floor is a state rather than a slogan.
- The meeting-2 demo (frozen merge → one command → plot, clean `git status`) carried to meeting 3 (28 Oct); hardening merges after supervisor sign-off, and the sweeps start 23 Oct.

## Phased plan

1. **Confirmation + CLI.** The additive `--seed` and the exit-2 config-error handler, then the frozen demo config across the 10 programme seeds, reported against the pre-registered Decision A/B split (seed-robust vs seed-sensitive).
2. **Size-independent data.** The `common_val` flag, the probe micro-change, the four dataset-size configs, and the n=64 pilot - the series' first point and the memorisation-gap hunt at small data.
3. **Instrumentation.** `tinylm/metrics.py`, the per-position and per-example sidecars, the per-run final checkpoint, and all-layer attention.
4. **Modular arithmetic.** The second task family: its Config fields and derivation, the ordered-pair split rule, and the two weight-decay pilots.
5. **Sweep tooling.** The runner and its registry, the two sweep figures, and the CSV rollup.
6. **Edge configs.** Odd-but-legal shapes, loud failures for illegal input, and a parametrized capacity ceiling over `dataclasses.replace` variants.
7. **Long-run stability + parity.** The 3000-step copy smoke, the 500- and 10 000-step mod-arith smokes, and the CPU/MPS parity check.

## Open questions - settled

1. Where the second task family lives: one `Config` with a `task` field and a dispatcher inside `generate_splits`, not a forked data module.
2. What "memorised" means: exact reproduction over the task's own span, threshold-free, so no ε has to be calibrated.
3. Whether checkpointing returns: only as a final-state artefact; resume stays parked.
4. Whether attention inspection returns: as all-layer output feeding a *between-run* snapshot; the series stays cut.
5. Where sweep evidence lives: `logs/` stays local, and canonical copies (registry, CSVs, PNGs) are committed under `docs/evidence/`.
6. Weight decay: scoped to mod-arith for the mechanism question, with the copy task held at 0.0.
7. How a run's provenance is recorded: no tag ritual, so the commit SHA is the anchor and each sweep row records the commit it ran from.

## Definition of done

- Every step green and floored; `uv run --frozen pytest` passes (67 tests).
- The copy trajectory value-identical to the frozen prototype merge except `elapsed` and `config_hash`.
- `configs/copy.yaml` byte-identical, and `metrics.jsonl` still exactly 10 columns.
- Canonical evidence committed under `docs/evidence/`.
- README and LOG current.
