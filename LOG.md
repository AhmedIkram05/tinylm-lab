# TinyLM Lab - Project Log

Dated, running log of meetings, decisions and milestones. Internal record;
meeting agendas are set by me.

## How this log is used (logging criteria)

This is a decision-and-meeting log, not a diary. Rule of thumb: an entry
exists because something changed state - would the supervisor care about it if
he read only this file before a meeting?

**Log:**

- Supervisor meetings: decisions, answers given, action items; next agenda.
- Decisions and direction changes, with rationale (the decision record for the
  project lives here).
- Milestones and working states: build checkpoints, first runs and what they
  showed, demo results.
- Failures that taught something (bug → cause → fix) - rigour evidence.
- Corrections and honesty fixes; standing notes (LSEP, confirmations, backups).

**Do not log:**

- Code changes, commits, file-level edits, dependency bumps - git owns those.
- Experiment metrics - `logs/*.jsonl` owns those at run time.
- Reading notes - Zotero annotations own those.
- Plan content - `docs/planning/plan.md` holds the plan; this file records
  decisions about it.
- Daily "worked on X" filler - time passing is not a state change.

**Convention:** append-only - past agendas convert into meeting records with
per-item outcomes (the plan-vs-outcome delta is informative); nothing is
deleted.

## Standing notes

- **Ethics/LSEP:** no human participants or personal data → no full ethics
  application expected; submit the LSEP declaration in Week 9.
- **Assessment timeline:** Interim report Week 13 = **formative feedback only, 0% of
  grade** (lit review + project plan, rubric feedback feeds the final
  report). Dissertation / final portfolio Week 24 Friday 12:00 = **100% of
  grade**. Demo Week 25, pass/fail (5 min demo + 10 min Q&A - must
  be able to explain own code). LSEP/ethics due Week 9. **Track chosen:
  primarily research with a substantial development side; formal declaration
  at submission (canvas Q1.2).** Research-track marks (the portfolio
  rubric): 25% RQ/rationale/lit review, 25% research design & rigour, 20%
  artefact design & implementation, 20% analysis & conclusions, 10%
  dissertation quality.
- **Backups:** code → GitHub; documents → OneDrive.

## 02-10-2026 - Meeting 1 (initial, Dr John Lawrence, ~10m)

Recorded scope and the first build target.

- **Scope:** the supervisor left the direction largely open, with the bar set
  high. Working standard adopted from this meeting: engineering depth
  over breadth - correct, tested, reproducible, inspectable work at every step.
- **Explicit answers given:** first build target = "have the Transformer built"
  (development only, no experiments yet); laptop-only compute (Apple
  Silicon/MPS), training runs ≤ a few minutes; mostly own implementation on
  PyTorch primitives (attention, embeddings, training loop); fortnightly
  meetings Wednesdays 13:30.
- **Decisions made:** first build target = tiny Transformer + minimal end-to-end
  pipeline (repeated-token copying task, seeded training loop, logged
  train/val metrics, one command → learning-curve plot); repo in separate dev
  folder, public on GitHub; full plan drafted and since consolidated into the
  repo as `docs/planning/prototype-spec.md`.
- **First task chosen:** repeated-token copying - simplest learnable structure,
  trains in seconds clean memorisation/generalisation split.

## 03-10-2026 - Direction set and docs consolidated

- **Direction reverted the narrowing:** all three RQ factors are mainline
  again - capacity, dataset size and task difficulty as one-factor-at-a-time
  series (10 seeds each) plus the grokking hunt on the modular-arithmetic
  thread, all on the same framework. Retired "headline question" / "Plan B"
  vocabulary; the dataset-size series is the **guaranteed floor** that answers
  the research question alone under any outcome.
- **Track:** primarily **research** with a substantial development side; formal
  declaration deferred to submission (canvas Q1.2 - supervisor left the track
  declaration open).
- **Budget corrected: 24+ hrs/week** (the earlier 8-10 hr estimate was wrong);
  phases re-checked against it.
- **Docs consolidated:** repo made the single source of truth - plan, prototype spec,
  ADRs and reading map live under `docs/`; OneDrive planning files retired.

## 04-10-2026 - Grokking deferral decision reversed; grokking confirmed mainline

- **ADR "grokking is stretch, not headline" deleted:** the feasibility record
  says laptop-scale is the grokking-literature norm - Power et al.'s grokking
  model (2-layer, width-128, 4-head decoder-only ≈ our spec defaults), Nanda's
  1-layer circuit analysis and Liu's same-scale transformers.
  The ADR's remaining content (priority discipline + guaranteed floor) is carried
  by plan.md. Remaining ADRs renumbered: 0001 laptop-first-compute, 0002 json-csv-logging-no-ui.
  Grokking confirmed mainline on the modular-arithmetic thread (as plan.md reflects),
  dataset-size series as guaranteed floor.
- **Honesty fix propagated:** "held-out patterns unpredictable by design" is
  now stated as the correct mechanism (README + prototype-spec): no pattern-to-pattern
  mapping exists, but a general positional copying rule could solve validation -
  memorisation is the expectation at this scale and budget, and a rising
  validation curve would be reported as a finding, not hidden.
- **Correction (fabricated attribution):** the meeting-1 record above stated
  "short project log + agenda sent beforehand" as a supervisor instruction.
  John never asked for this (flagged by Ahmed, 2026-10-04); the phrase
  originated in our own tooling, not the meeting. Removed from the meeting-1
  record here and from README/plan.md. The log is an internal decision-and-
  meeting record; agendas are set by Ahmed and shared at his discretion.

## 08-10-2026 - Prototype Complete (Transformer + pipeline + tests)

Target state: the "have the Transformer built" target from meeting 1 - `tinylm/` (utils, data, model, train), `configs/copy.yaml`, 17 pytest assertions green including the CPU bit-exact reproduction test. One command → metrics + graph/plot.

- **Built:** `Config` (17 fields, validated) + YAML loader + seeds/device helpers; period-4 sampler (512/512 disjoint patterns, purity-rejected, fixed 128 probe, seeded Torch batcher); hand-written causal Transformer (sinusoidal PE, pre-LN, weight-tied output head, 397,824 parameters); training loop (pinned AdamW, LM-shift loss, copyable-position accuracy, JSONL + Okabe-Ito plot).
- **Failures that taught something:** the first training run exposed an accuracy-slice off-by-one bug: `targets[:, 3:]` produced 29 target positions versus 28 prediction positions; this was fixed by aligning the slice with `targets[:, period:]`. The overfit test initially passed raw `(B,T,V)` logits to `cross_entropy`; this was a test-only bug, as the training implementation already handled the tensor dimensions correctly.
- **Honest finding:** an MPS training run reached `train_acc 1.0000 / val_acc 0.9999` - at this budget, the results are consistent with the model learning the general copying rule rather than merely memorising the training patterns, but this is not yet established. The memorisation-vs-rule question therefore remains open for the experiment phase, with tougher settings such as smaller datasets and period sweeps.

## 08-10-2026 - CPU canonical, MPS opt-in only

- **Decision**: canonical training backend is CPU (`configs/copy.yaml device: cpu`); MPS is opt-in, `--device mps`. MPS is faster on the current 0.4M tranformer but CPU is still chosen for determinism and CI compatibility, *not* speed: speed bump at this current size is minutes across the whole experiment phase, while MPS's nondeterminism would void the bit-exact same-seed guarantee the seed protocol depends on.
- **Why:** `get_device("auto")` preferred MPS on this Mac, so every real run was MPS while `test_cpu_repro` proved bit-exactness on CPU - a split-brain (guarantee certified a path never used). Now main, floor run, repro test and `auto` itself are all CPU; the thing run is the thing proven deterministic.
- **Changed:** `configs/copy.yaml` (`auto` -> `cpu`, one-time re-anchor); `get_device("auto")` now returns CPU, with MPS reachable only via `--device mps`; README Reporting line.

## 13-10-2026 - Research question frozen (main RQ + sub-question)

**Main RQ:** How do model capacity, dataset size and task difficulty affect memorisation and generalisation in miniature Transformers trained on synthetic sequence-copying and modular-arithmetic tasks?

**Sub-question:** Under what conditions does delayed generalisation (grokking) emerge on the modular-arithmetic task?

- **Objectives carry the trimmed detail** (graded in the rationale/method, not the RQ): significance = alignment with large-scale scaling research (both outcomes are findings); measurement = train/val performance with the headline gap = training-minus-validation accuracy; conditional mechanism aim (only if the work goes beyond training curves + attention maps). The two task families (copy, mod-arith) each carry their own difficulty control and differ in kind, not on one scale.
- **Source + supersession:** the canvas is the RQ's origin; this wording supersedes the earlier RQ phrasing in `plan.md`, `README.md` and `literature-review-plan.md` (header RQ + §1.2/§1.3/§7.5/§8.4), which now mirror it. Wording only - no experimental change.
- **Open with supervisor:** confirm a sub-question is permitted (in-repo guidance asks for the RQ in one sentence; it does not forbid sub-questions). **Resolved 14 Oct:** John confirmed sub-questions are permitted - see the Meeting 2 record below.

## 14-10-2026 - Meeting 2 did not take place; written update + email reply

No meeting: the supervisor was not in his office at 13:30 (locked); Ahmed
waited ~15 minutes. A written update was sent by email the same day and the
supervisor replied. Next meeting: **Wed 28 Oct 13:30** , was my fault for not
confirming the meeting time/day beforehand with John.

- **Supervisor expectation (heard):** John had expected the meeting to be
  confirmed beforehand ("I had thought that we would confirm before the
  meeting"). Meetings are confirmed in advance from now on.
- **Written update sent (14 Oct):** prototype description (~0.4M-parameter
  scratch Transformer; one command → train → metrics → plot); `learning_curve.png`
  attached; repo link (github.com/AhmedIkram05/tinylm-lab); Gantt chart
  attached; the RQ proposed for freeze; a proposed sub-question; and a
  next-steps priority (hardening, then experiments).
- **Explicit answers (John, by reply):** the RQ "looks really good" - **RQ
  approved as written**; **a secondary RQ is permitted** and the proposed
  sub-question "sounds good" (multiple RQs are fine if complementary and
  answerable in the project time); looking forward to 28 Oct.
- **Agenda items → outcomes:**
  1. Demo - **not demoed live** (no meeting). Plot + repo link sent by email;
     live demo carries to 28 Oct.
  2. Canvas gap / RQ sign-off - **RQ signed off** by email (main +
     sub-question); the experimental plan itself was not discussed.
  3. LSEP declaration form - **not raised in the email; still open.** Ask
     where it is distributed on 28 Oct (Week 9 deadline).
  4. Whole-project plan / Gantt - Gantt sent by email with link to GH repo.
- **Carried to 28 Oct:** LSEP form source; experiments/sweeps opinion.

## Agenda - Meeting 3 (Wed 28 Oct 13:30)

1. Demo: live run - one command → plot; seed-reproducibility test passing;
   `git status` clean. (Carried from meeting 2.)
2. LSEP declaration form - where is it distributed? (Week 9 deadline.)
3. Next-steps priority - confirm harden → experiments (decided: hardening ends
   22 Oct; sweeps start 23 Oct, so this meeting is a mid-sweep checkpoint -
   feedback folds into the iteration buffer).
