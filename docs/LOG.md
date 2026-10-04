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

## 2026-10-02 - Meeting 1 (initial, Dr John Lawrence, ~10m)

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
  repo as `docs/planning/m1-spec.md`.
- **First task chosen:** repeated-token copying - simplest learnable structure,
  trains in seconds clean memorisation/generalisation split.

## 2026-10-03 - Direction set and docs consolidated

- **Direction reverted the narrowing:** all three RQ factors are mainline
  again - capacity, dataset size and task difficulty as one-factor-at-a-time
  series (≥3 seeds each) plus the grokking hunt on the modular-arithmetic
  thread, all on the same framework. Retired "headline question" / "Plan B"
  vocabulary; the dataset-size series is the **guaranteed floor** that answers
  the RQ alone under any outcome (CONTEXT.md).
- **Track:** primarily **research** with a substantial development side; formal
  declaration deferred to submission (canvas Q1.2 - supervisor left the track
  declaration open).
- **Budget corrected: 24+ hrs/week** (the earlier 8-10 hr estimate was wrong);
  phases re-checked against it.
- **Docs consolidated:** repo made the single source of truth - plan, prototype spec,
  ADRs and reading map live under `docs/`; OneDrive planning files retired.

## 2026-10-04 - Grokking ADR retired; grokking confirmed mainline

- **ADR "grokking is stretch, not headline" deleted:** the feasibility record
  says laptop-scale is the grokking-literature norm - Power et al.'s grokking
  model (2-layer, width-128, 4-head decoder-only ≈ our spec defaults), Nanda's
  1-layer circuit analysis and Liu's same-scale transformers; verified during
  the literature grill. The ADR's remaining content (priority discipline +
  guaranteed floor) is carried by CONTEXT.md and plan.md. Remaining ADRs
  renumbered: 0001 laptop-first-compute, 0002 json-csv-logging-no-ui. Grokking
  confirmed mainline on the modular-arithmetic thread (as plan.md reflects),
  dataset-size series as guaranteed floor.
- **Honesty fix propagated:** "held-out patterns unpredictable by design" is
  now stated as the correct mechanism (README + m1-spec): no pattern-to-pattern
  mapping exists, but a general positional copying rule could solve validation -
  memorisation is the expectation at this scale and budget, and a rising
  validation curve would be reported as a finding, not hidden.
- **Correction (fabricated attribution):** the meeting-1 record above stated
  "short project log + agenda sent beforehand" as a supervisor instruction.
  John never asked for this (flagged by Ahmed, 2026-10-04); the phrase
  originated in our own tooling, not the meeting. Removed from the meeting-1
  record here and from README/plan.md. The log is an internal decision-and-
  meeting record; agendas are set by Ahmed and shared at his discretion.

## Agenda - Meeting 2 (Wed 14 Oct 13:30 - confirmed)

1. Demo: prototype live run - one command → plot; seed-reproducibility test
   passing; attention-weight inspection if time allows. Reading programme
   complete: 17 sources, annotated in Zotero - interim report, literature
   review is assembly-ready.
2. Canvas gap to close: when to bring the finalised RQ + experimental plan
   for sign-off.
3. LSEP declaration form: we don't have it - ask where it's distributed
   (MyDundee module area?) so it's ready before the Week 9 deadline.
4. Whole-project plan / Gantt chart (W1 slides: "Agree your own plan with your
   supervisor"): **drafted - see `docs/planning/plan.md`** - module milestones
   fixed (LSEP W9, interim W13, portfolio W24, demo W25), project phases
   mapped to the marking criteria, risks + guaranteed floor stated; doubles as
   interim-report "project plan" material; LOG.md = time-actually-spent
   record.

## Standing notes

- **Ethics/LSEP:** no human participants or personal data → no full ethics
  application expected; submit the LSEP declaration in Week 9.
- **Assessment timeline (weighting per Ahmed's module info - confirmed):**
  Interim report Week 13 = **5% of grade** (lit review + project plan,
  rubric feedback feeds final report). Final portfolio Week 24 Friday 12:00
  = **95% of grade**. Demo Week 25, pass/fail (5 min demo + 10 min Q&A - must
  be able to explain own code). LSEP/ethics due Week 9. **Track chosen:
  primarily research with a substantial development side; formal declaration
  at submission (canvas Q1.2).** Research-track marks:
  25% RQ/rationale/lit review, 25% research design & rigour, 20% artefact
  design & implementation, 20% analysis & conclusions, 10% dissertation
  quality. NOTE: W1 welcome slides describe the interim report as "formative"
  - generic deck, treated as wrong. Weighting confirmed directly with John
  (2026-10-02): interim **5%**, final portfolio incl. demo **95%**.
- **Backups:** code → GitHub; documents → OneDrive.
