# TinyLM Lab — Project Log

Dated, running log. Sent (or linked) to the supervisor before each fortnightly
meeting together with the next meeting's agenda.

## 2026-10-02 — Meeting 1 (initial, Dr John Lawrence)

Recorded scope and first milestone.

- **Scope:** everything "up to me" — permissive on direction; the stated bar is
  "insanely impressive at every step". Interpreted as engineering depth
  (correct, tested, reproducible, inspectable), not breadth.
- **Explicit answers given:** Milestone 1 = "have the Transformer built"
  (development only, no experiments yet); laptop-only compute (Apple
  Silicon/MPS), training runs ≤ a few minutes; mostly own implementation on
  PyTorch primitives (attention, embeddings, training loop); fortnightly
  meetings Wednesdays 13:30, short project log + agenda sent beforehand.
- **Decisions made:** Milestone 1 = tiny Transformer + minimal end-to-end
  pipeline (repeated-token copying task, seeded training loop, logged
  train/val metrics, one command → learning-curve plot); repo in separate dev
  folder, public on GitHub; full plan drafted in
  `Planning/Milestone 1 - Framework v1.md`.
- **First task chosen:** repeated-token copying — simplest learnable structure,
  trains in seconds, clean memorisation/generalisation split.

## Agenda — next meeting (Wed 14 Oct 13:30, to confirm)

1. Demo: Milestone 1 live run — one command → plot; seed-reproducibility test
   passing; attention-weight inspection if time allows.
2. Canvas gaps to close: build-vs-reuse level confirmation (canvas Q3.1);
   when to bring the finalised RQ + experimental plan for sign-off (Q8.3).
3. Plan to Framework v1 (Week 10): checkpointing, config library, second task.

## Standing notes

- **Ethics/LSEP:** no human participants or personal data → no full ethics
  application expected; submit the LSEP declaration in Week 9.
- **Backups:** code → GitHub (public); documents → OneDrive.
