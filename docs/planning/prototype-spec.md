# Prototype and pipeline spec - "the Transformer is built"

**Status:** Completed (08 Oct 2026)

**Goal:** By the next supervisor meeting (Wed 14 Oct 13:30) build a tiny Transformer LM from scratch on PyTorch primitives, plus a minimal end-to-end pipeline: one synthetic task → fixed-seed training loop → logged train/val metrics → one command → one plot, tested and reproducible.

## Constraints

- Supervisor milestone: "have the transformer built": development only, no experiments yet.
- Laptop-only (Apple Silicon, CPU canonical): 0.4M parameters  runs take seconds . The backend is not the constraint - the project keeps every variant under the ≤2M ceiling, because the result is more impressive/stronger at small scale (demo ≈ 0.4M).
- Own implementation on torch primitives (attention, embeddings, loop) - no `nn.Transformer`, no copied reference.
- Engineering depth over breadth: correct, tested, reproducible, inspectable; extras will go to the parking lot.

## Spec

- **Model** (`tinylm/model.py`): √d-scaled embeddings + hand-written sinusoidal PE (zero params, exact-value test); hand-written causal MHA with `return_attention`; GELU-exact FFN; pre-LN blocks + final LN; tied head, no flag; init normal(0, 0.02) for embeddings/head, torch default for linears; flat 17-field `Config`; `device: cpu` canonical (bit-exact); `auto` resolves to CPU; `--device mps` is opt-in parity only; demo ≈ 0.4M, every capacity variant stays under the ≤2M ceiling.
- **Task** (`tinylm/data.py`): repeat-copying, a 4-token pattern repeated to `seq_len=32` over 8 tokens; the sampler rejects impure periods and draws seeded 512 train / 512 val patterns from disjoint sets (disjointness asserted; every token appears in training); causal-LM shift targets; fixed `torch.long` (B, T) batches; eval probes stay fixed (128-pattern train subset + full val set, never the live batch).
- **Training** (`tinylm/train.py`): AdamW values pinned at lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, constant LR; defaults d_model=128, n_heads=4, n_layers=2, d_ff=512, batch 32, 500 steps, eval every 25; loss is mean CE over shifted positions, accuracy argmax match over copyable positions (4..31); eval under `torch.no_grad()` on fixed probes; JSONL logs 10 columns (step, train/val loss/acc, lr, elapsed, seed, config_hash, torch_version); seeds fixed with deterministic algorithms where allowed, dropout 0.0 wired-off; CLI `python -m tinylm.train` writes `logs/<run>/metrics.jsonl` + `learning_curve.png` (`--config` → `configs/copy.yaml`, `--run-name` → `copy-<timestamp>`, never overwrite); one figure, two panels, Okabe-Ito, chance baseline at 1/vocab; deps torch, numpy, matplotlib, pyyaml, pytest, py>=3.11, uv.lock pinned.
- **Config** (`configs/copy.yaml`) stays exactly as below:

```yaml
seed: 0
vocab_size: 8
seq_len: 32
period: 4
n_train: 512
n_val: 512
d_model: 128
n_heads: 4
n_layers: 2
d_ff: 512
dropout: 0.0
lr: 0.001
weight_decay: 0.0
batch_size: 32
steps: 500
eval_every: 25
device: cpu
```

- **Utils** (`tinylm/utils.py`): holds `set_seeds`, `get_device`, yaml → `Config` loading. Tiny by design.
- **test_model.py** checks output shapes, causal masking (zero weights + logit invariance), return_attention shape, tied weights, param ceiling.
- **test_data.py** checks seed determinism, split disjointness, period purity, vocab coverage.
- **test_train.py** checks an overfit-one-batch smoke plus a yaml → Config round-trip.
- **test_repro.py** checks two CPU subprocess runs log identical rows except `elapsed`; CPU scope is stated in the README.
- **Layout**: `tinylm/__init__.py`, `tinylm/model.py`, `tinylm/data.py`, `tinylm/train.py`, `tinylm/utils.py`, configs/copy.yaml, tests/{test_model,test_data,test_train,test_repro}.py, pyproject.toml + uv.lock, README.md, .gitignore, LICENSE (MIT), docs/ (LOG.md, planning/, adr/, reading-list-seeds.md).
- **Meeting 2 (Wed 14 Oct)**: runs one command live and the plot appears; shows LOG.md, reproduction test; and asks any burning Qs.

## Parking lot

| Idea | Parked until |
| --- | --- |
| Checkpointing + config library | Experiment phase, if added to scope |
| Multi-task registry | Experiment phase, only if a second task is ever justified, unlikely |
| Web dashboard / W&B / distributed anything | Indefinitely |

## Timeline

- Sat 3 - Mon 5 Oct: workshop tasks + development skeleton.
- Tue 6 - Thu 8 Oct: model → task → training loop → tests, in that order; freeze proto-v1 on 8 Oct.
- Fri 9 - Wed 14 Oct: demo polish from the frozen build, meeting 2 agenda/Qs, LOG.md entry; hardening runs in parallel (started 9 Oct) and merges after supervisor sign-off.

## Phased plan

1. Skeleton: pyproject, lockfile, gitignore, package dirs; checks uv sync + torch import.
2. Utils + config: flat Config, yaml loader, seeds, device (cpu canonical; auto resolves to cpu; mps opt-in).
3. Data: seeded sampler with purity rejection, disjoint 512/512, 128 probe, batcher.
4. Model: sinusoidal PE, causal MHA + return_attention, GELU exact, pre-LN, tied head, init plan.
5. Train: AdamW plan, shift loss, copyable acc, probe eval, JSONL, CLI, 2-panel plot.
6. Tests: the four files above; each goes green before the next step starts.
7. Polish: README, LOG entry, one-command demo, push.

## Open questions - settled

1. PE is sinusoidal, hand-written, exact-value test; learned PE stays later work.
2. Task is period-4, vocab 8, 512/512 disjoint, seq 32; disjoint 4-tuples give the size knob.
3. Dropout is 0.0, wired but off; overfitting is the signal of interest.
4. Embeddings are tied, no flag; untied stays an experiment-phase question.
5. Defaults are d128 / h4 / L2 / d_ff512, AdamW 1e-3, batch 32, 500 steps, eval 25.
6. Inspection comes from return_attention from day one; the one-command plot is the demo blocker.

## Definition of done

- One command reproduces train → evaluate → plot from a saved config.
- pytest passes locally incl. CPU reproduction test.
- Same seed gives same logged CPU metrics.
- README explains the 10-minute run down.
- LOG.md entry written.
