# Prototype and pipeline spec - "the Transformer is built"

**Status:** Ready to begin implementation

**Goal (one line):** By the next supervisor meeting (Wed 14 Oct 13:30): a tiny Transformer LM implemented from scratch on PyTorch primitives, plus a minimal end-to-end pipeline - one synthetic task → seeded training loop → logged train/val metrics → one command → one plot - tested and reproducible.

---

## Source constraints (supervisor meeting + W1 canvas)

- Supervisor's milestone scope: **"have the transformer built"** - development only, no experiment design yet.
- Laptop-only compute (Apple Silicon, MPS backend). Training runs ≤ a few minutes. Models ≤ ~2M params (default config ≈ 0.4M; the headroom exists to extend the capacity-sweep axis, not to grow the default).
- **Mostly own implementation**: attention, embeddings, training loop written on PyTorch primitives - not `nn.Transformer`, not a copied reference implementation (nanoGPT/Karpathy is reading material, not copy material).
- Working standard: **engineering depth** (correct, tested, reproducible, inspectable), not breadth. Anything shiny → parking lot.
- This build is the dev-start of a prototype: configurable tiny Transformer, fixed seeds, logged train/val metrics, one plot, one-command reproducibility.

---

## Spec

### Model (`tinylm/model.py`)

- Token embeddings + **sinusoidal** positional encodings (decided: hand-written formula from Vaswani, zero params, exact-value unit test). Token embeddings scaled by √d_model before the PE is added (Vaswani's choice, pinned).
- Causal multi-head self-attention: hand-written scaled dot-product (Q/K/V projections, causal mask). Contract: `model(x)` returns logits; `model(x, return_attention=True)` returns `(logits, attn)` with `attn` shape `(B, n_heads, T, T)` → inspectable evidence, enables heatmap. Dropout, when wired on, applies to attention weights only.
- Feed-forward network (d_model → 4·d_model → d_model), GELU exact (`approximate='none'`).
- Pre-LayerNorm residual blocks × n_layers, plus a final LayerNorm before the head.
- Output head → next-token logits (**tied** embeddings, decided - no config flag; untied is an experiment-phase question if it ever matters).
- Weight init: embeddings and head `normal(0, 0.02)`; Linear layers PyTorch default (kaiming_uniform). Pinned in code, named in the config.
- One flat `Config` dataclass covering model + data + training: `vocab_size, seq_len, period, n_train, n_val, d_model, n_heads, n_layers, d_ff, dropout, lr, weight_decay, batch_size, steps, eval_every, seed, device` - everything important lives in the yaml (schema shown below).
- Demo config ≈ 0.4M params (ceiling ~2M). `device: auto` = mps → cpu fallback; `cpu`/`mps` force a backend.

### Task (`tinylm/data.py`) - decided: period-4, 512/512

- Repeated-token copying: sequences are a 4-token pattern repeated to `seq_len=32` (8 repeats), over a vocab of 8 pattern tokens → 8⁴ = 4096 possible patterns. Sampled 4-tuples whose true period is < 4 (e.g. AAAA, ABAB) are rejected at generation time - period purity keeps the task-difficulty axis clean.
- Seeded generator (numpy Generator, seed from config): sample **512 train patterns / 512 val patterns from disjoint sets** (assert disjointness). This gives the dataset-size series its knob - sweepable (64/256/512/1024) in the experiment phase.
- "Novel combination" is exact: the two splits share no 4-tuple content (asserted disjointness; every token appears in training - also asserted in tests). The open question is the within-sequence positional rule: a general copy rule (token t−4) or in-context induction would solve val perfectly, but no val pattern can be derived from any train pattern. At this model size and budget we EXPECT memorisation, and either outcome is reportable: train↓ / val-flat is consistent with memorisation without generalisation (not proof of mechanism); val rising means the model found the general copying rule, itself a finding. Stated honestly in README.
- Causal-LM framing (input = sequence, target = shifted by one); fixed-length batches, no padding needed.
- Batches are `torch.long`, shape (B, T), moved to the active device per batch.

### Training (`tinylm/train.py`)

- AdamW + cross-entropy. **Pinned:** `AdamW(lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)` - wd pinned at 0.0 here; nonzero wd belongs to the regularisation thread, not the prototype. Constant LR, no schedule, no gradient clipping.
- **Defaults (decided):** d_model=128, n_heads=4, n_layers=2, d_ff=512 (≈0.4M params); batch 32, 500 steps, eval every 25 steps (20 log points) - target: visibly learning in <2 min on MPS, with headroom.
- Batch sampling: each step draws `batch_size` pattern indices uniformly with replacement from a `torch.Generator` seeded with `config.seed`, indexing the fixed 512×32 train tensor.
- Loss/metric definitions: loss = mean cross-entropy over positions 1..31 with the LM shift (`logits[:, :-1]`, `targets[:, 1:]`). Accuracy = mean per-position argmax match over the copyable positions (4..31) - the first 3 targets have no in-context evidence and are irreducibly unpredictable.
- Eval every N steps, in eval mode under `torch.no_grad()`: train metrics on a fixed train-probe subset (128 train patterns, fixed at data-gen time) and val metrics on the full 512-pattern val set - never the current training batch (recency-biased, uncomparable across dataset sizes).
- Log per eval interval to JSONL: `(step, train_loss, train_acc, val_loss, val_acc, lr, elapsed, seed, config_hash, torch_version)`. Bit-exact comparison covers the numeric loss/acc/lr columns only; `elapsed` is wall-clock and excluded by definition.
- Seeds fixed at start: `random`, `numpy`, `torch`; `torch.use_deterministic_algorithms(True)` where the op set allows. Dropout is only invoked when p > 0 (wired-but-off dropout must not consume RNG). The CPU repro test is scoped to one machine and thread count (MPS determinism caveat below).
- CLI: `python -m tinylm.train` → trains → writes `logs/<run>/metrics.jsonl` + `learning_curve.png`. `--config` defaults to `configs/copy.yaml` (sweeps pass others); `--run-name` defaults to `copy-<YYYYMMDD-HHMMSS>` and run dirs never overwrite. Plot: one figure, two panels (loss, accuracy), Okabe-Ito colour-blind-safe palette, dashed chance baseline at 1/vocab.
- Dependencies (pyproject): `torch, numpy, matplotlib, pyyaml, pytest`; `requires-python >= 3.11`; `uv.lock` pins exact versions.

### Example config (`configs/copy.yaml`)

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
device: auto
```

### Utils (`tinylm/utils.py`)

- `set_seeds(seed)`, `get_device()`, config loading (yaml → one flat `Config` dataclass). Keep tiny.

### Tests (`tests/`)

- `test_model.py` - output shapes (B,T,V); causal masking correct: masked attention weights = 0 AND logit invariance (changing future tokens leaves earlier-position logits unchanged); `return_attention=True` returns (B, n_heads, T, T); tied weights are literally the same tensor; param count ≈ 0.4M at defaults with a hard ceiling of 2M; demo config builds a valid model.
- `test_data.py` - generator deterministic under seed; split disjoint; every sampled pattern has true period = 4; all 8 vocab tokens appear in training; val sequences rule-consistent and novel; batches are torch.long (B,T).
- `test_train.py` (smoke) - overfit-one-batch: 50 steps on a single fixed batch collapses loss (catches shift/mask/loss wiring bugs end-to-end); yaml → Config loader round-trip on configs/copy.yaml.
- `test_repro.py` - two separate CPU subprocess runs (`--steps 50 --device cpu`) with the same seed produce identical logged rows on all columns except `elapsed`. **Honest caveat:** MPS may not be bit-deterministic across runs/drivers - the guarantee is pinned on CPU and stated as such in the README.

### Repo layout

```text
tinylm-lab/
  tinylm/{__init__,model,data,train,utils}.py
  configs/copy.yaml
  tests/{test_model,test_data,test_train,test_repro}.py
  pyproject.toml  uv.lock
  README.md  .gitignore  LICENSE (MIT)
  docs/ (LOG.md, planning/, adr/, reading-list-seeds.md)
```

### Meeting demo (Wed 14 Oct)

- Live: run the one command → plot appears.
- Show LOG.md + agenda, repro test green, attention heatmap if time allows.
- Scope of the claim: the demo curve demonstrates learning plus a signature consistent with memorisation - not overfitting dynamics and not grokking; the pre-registered grokking criteria live in the later experiment specs.

---

## Parking lot (decisions for later)

| Idea | Parked until |
| --- | --- |
| Checkpointing + config library | Experiment phase - promotable inside the corrected 24+ hrs/week budget; supervisor promotes per CONTEXT.md |
| Multi-task registry | Experiment phase - only if a second parallel task is ever justified |
| Web dashboard / W&B / distributed anything | Indefinitely |

(Modular-arithmetic task and the seeds-≥3 / dataset-size protocol left the parking lot on the 2026-10-03 direction change - now mainline in `docs/planning/plan.md`.)

## Timeline (~15-18 hrs minimum before 14 Oct, inside 24+ hrs/week)

- **Sat 3 - Mon 5 Oct:** workshop tasks (repo, project log, Zotero, reading start) + dev skeleton.
- **Tue 6 - Mon 12 Oct:** model → task → training loop → tests, in that order.
- **Tue 13 - Wed 14:** polish demo, agenda, log entry.
- If it slips: cut eval granularity, not the transformer - the transformer is the core deliverable of this build.

## Open questions - RESOLVED

1. **Positional encoding: sinusoidal.** Hand-written formula, exact-value test, zero params; learned is 10 lines if the experiment phase ever needs it (YAGNI now).
2. **Copy-task shape: period-4, vocab 8, 512/512 disjoint split, seq_len 32.** Wins because the dataset-size series needs a sweepable knob - period-2's 64 pairs would have dead-ended it. Novel combination = disjoint 4-tuple sets.
3. **Dropout: 0.0, wired but off.** Overfitting is the phenomenon of interest; dropout would've confound the demo.
4. **Tied embeddings: yes, no flag.** Untied is an experiment-phase question.
5. **Defaults:** d_model=128, n_heads=4, n_layers=2, d_ff=512 (≈0.4M); AdamW 1e-3, batch 32, 500 steps, eval 25.
6. **Heatmap:** `return_attention` built + causal-masking-tested (it's the inspectability evidence); heatmap script = optional, if time allows; demo blocker is the one-command plot.
7. **Tooling: uv + pyproject.toml** with pinned lockfile - makes the CPU seed-repro claim verifiable (torch version pinned).
8. **Package name: `tinylm`** (repo `tinylm-lab`). Settled.

## Definition of done

- One command reproduces train → evaluate → plot from a saved config.
- `pytest` passes locally (incl. CPU seed-reproducibility).
- Same seed → same logged metrics (CPU).
- README explains the 10-minute hello-world.
- LOG.md entry written; repo pushed to GitHub.
