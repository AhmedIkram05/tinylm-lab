# TinyLM Lab - Whole-Project Plan

**Status:** Live - prototype-spec.md completed (08 Oct) and prototype-hardening.md completed (22 Oct); experiment phase next, sweeps start 23 Oct

**Research question:** How do model capacity, dataset size and task difficulty affect memorisation and generalisation in miniature Transformers trained on synthetic sequence-copying and modular-arithmetic tasks?

**Sub-question (secondary aim):** Under what conditions does delayed generalisation (grokking) emerge on the modular-arithmetic task?

> The experiment programme runs all three factors as mainline one-factor-at-a-time series (10 seeds each) with the grokking sub-question riding the modular-arithmetic thread, all on the same framework. The task set is two families, each with its own difficulty control - sequence copying (difficulty: period 2 → 4 → 8) and modular arithmetic (difficulty: train_frac and/or modulus size) - which differ in kind (different computational behaviour), not as points on one universal difficulty scale. Bar: maximally impressive at every step - effort is not the constraint. The dataset-size series alone still answers the RQ with the same framework, so the programme is guaranteed deliverable under any outcome.

---

## Module Assessed Deliverables

| Module week | Deliverable | Weight |
| --- | --- | --- |
| Week 9 | Ethics / LSEP declaration form | 0% |
| Week 13 | Interim report (literature review + project plan) | formative feedback only (0%) |
| Week 24 (Fri 02 Apr 12:00) | Final portfolio - written report 15-35 pages incl. references + artefact codebase + appendices (ethics materials, visuals, engineering docs) | 100% |
| Week 25 (≈ Fri 30 Apr, exam weeks 25-28) | Demo: 5 min + 10 min Q&A | pass/fail |

Supervisor meetings: fortnightly, Wednesdays 13:30.

## Phase plan

| Phase | When (calendar) | Output |
| --- | --- | --- |
| Framework core - Transformer + minimal pipeline (early build) | 3 Oct - 8 Oct (6d) | One command → train → metrics → plot; tests green incl. CPU seed repro. |
| Hardening - methods freeze + the code the sweeps need | 9 Oct - 22 Oct (done) | Machine-triageable CLI (exit 2 on any config error); a size-independent common-val plus the dataset-size series; an in-loop per-example memorisation measurement; the modular-arithmetic task; the sweep runner, figures and CSV rollup with committed evidence; edge cases and the capacity ceiling. Merges after supervisor sign-off; sweeps start 23 Oct. Contingency is the iteration buffer to 30 Nov. |
| Reading & literature grounding | 24 Sep - 18 Dec (ongoing, feeds interim report, due 18 Dec) | sources collected in Zotero; page-anchored annotations from 8 Oct; synthesis (theme statements + gap) late Oct - mid Nov. |
| Experiments - capacity × dataset size × task difficulty + grokking | Core sweeps 23 Oct - 13 Nov (starts after hardening ends 22 Oct); iteration buffer + grokking extension to 30 Nov | One-factor-at-a-time series × 10 seeds on the same framework, mapped onto the RQ's three factors: **capacity** (width d064/d128/d256 and depth l1/l2/l4, run on *both* task families), **dataset size** (the copy series at 7 log-spaced points 16-1024, plus the mod-arith `train_frac` axis), **difficulty** (copy period 002/004/008; mod-arith modulus 7/11/13) - with a **regularisation** (weight-decay) thread as an added fourth, and grokking riding the modular-arithmetic thread with a WD=0 control. The two task families differ in kind, not on one difficulty scale; the copy series uses the fixed common-val (the period / `train_frac` / modulus series cannot share one, and say so); ≈330 runs, whole programme ≈2 h on one laptop; gap-vs-factor comparisons (headline gap: training-minus-validation accuracy); honest failure log; any stalling thread → decision with supervisor |
| Interim report (Week 13) | late Nov - mid Dec | Project plan (written) + first lit-review synthesis draft (context / themes / closest work / gap) - dull but needed, synthesis work; formative feedback only (0% grade) |
| Testing & evaluation | Jan - mid Feb (continuous) | Longer and additional runs where the analysis needs them (edge cases and CPU/MPS parity were settled during hardening, not here); sweep analysis: gap-vs-factor plots across every series, 10-seed error bars, limitations; negative results reported honestly |
| Dissertation writing | late Jan - early Apr (to W24 Fri 02 Apr; overlaps testing to 12 Feb - analysis as runs land) | Report + appendices (configs, logs, plots) |
| Final portfolio (Week 24) | Fri 02 Apr 12:00, Week 24 | 100% of grade |
| Demo prep + delivery (Week 25) | 19-30 Apr (≈ Fri 30 Apr delivery; spans pre-exam + exam weeks 25-28) | 5-min demo + Q&A; every submitted claim explainable live; prep time-boxed, no new claims |

## How the plan maps to the marking criteria (research track)

| Criterion | Where the plan earns it |
| --- | --- |
| RQ, rationale, literature (25%) | Reading programme + interim lit review + rationale for the full three-factor RQ |
| Research design & rigour (25%) | Seeded, tested, reproducible pipeline; 10 seeds per config; guaranteed floor (dataset-size series); honest reporting |
| Artefact design & implementation (20%) | Framework: own-implementation Transformer, one-command reproducibility, inspectable attention |
| Analysis & conclusions (20%) | Three-factor sweep analysis (capacity × dataset size × task difficulty), gap quantification, grokking evidence (or bounded non-observation, reported honestly), negative results as findings |
| Dissertation quality (10%) | Writing window W18-24 (to Fri 02 Apr; overlaps testing 25 Jan-12 Feb intentionally - analysis as runs land), appendices generated by the framework itself |

Every criterion's upper bands (both portfolio rubrics) explicitly reward legal, social, ethical and professional treatment, and the interim-report rubric has a dedicated "Professional considerations" criterion. The section below is the standing LSEP record the dissertation and interim report draw on - kept current as the project runs.

## Legal, social, ethical and professional considerations (LSEP)

- **Legal:** the artefact is own implementation on PyTorch primitives; PyTorch (BSD-style licence) is the only third-party dependency, so there is no licence contamination in the shipped code. The repo is MIT-licensed and public; nothing is copied, and all data is generated in-house - no reuse rights or data-protection law engaged at all.
- **Social / human impact:** synthetic data only - no human participants and no personal data of any kind, by construction. Week 14's "user engagement" stage is therefore inapplicable and no full ethics application is expected; the LSEP declaration is submitted in Week 9. Because every datapoint is generated from documented rules, conclusions cannot overstate what a model learned from unexamined data - a common failure mode in applied ML that this design rules out.
- **Environmental:** the entire programme runs on one laptop - models ≤ ~2M parameters, runs of minutes, not GPU-hours. The compute footprint is negligible (documented in the appendices), and no cloud or university GPU time is a hidden dependency.
- **Honest reporting (research integrity):** all runs are logged including failed and negative ones; every claim in the report is backed by a logged metric or an inspection artefact; results are reported with 10-seed spread and limitations named. "Grokking did not appear at this scale" is a finding, not a failure - the guaranteed floor (dataset-size series) answers the RQ regardless. README and write-up state the held-out mechanism honestly: no pattern-to-pattern mapping exists, but a general copying rule could solve validation - memorisation is the expectation, and a rising validation curve would be reported as a finding, not dressed up or hidden.
- **Professional practice:** reproducible by construction (pinned lockfile, fixed seeds, bit-exact CPU repro test); accessible, non-proprietary outputs (JSONL/CSV logs, colour-blind-safe matplotlib plots); every submitted claim explainable live at the demo - the ownership-and-authorship bar is met by design, not by assertion.

## Constraints the plan respects

- **Laptop-first:** Apple Silicon laptop, CPU canonical (bit-exact + CI-compatible; MPS measures faster but is nondeterministic, so it is opt-in only). Capacity headroom extends the capacity-sweep axis but stays under the ≤2M ceiling - a deliberate choice (the result is more impressive at small scale), not a backend limit.
- **Scope discipline:** the parking lot holds tooling and UI only (dashboard, W&B, distributed anything); it never holds experiment series - all three RQ factors run.
- **Guaranteed floor (formerly Plan B):** the dataset-size series alone answers the RQ with the same framework, same plots - whatever happens to the other threads.

## Known risks

| Risk | Mitigation |
| --- | --- |
| Grokking doesn't appear within the experiment window | The three sweep series run regardless on the identical framework - no time lost; bounded non-observation reported honestly |
| MPS nondeterminism | MPS measures faster than CPU but is nondeterministic, so the bit-exact guarantee is pinned on CPU (`test_cpu_repro`); MPS stays opt-in. The hardening parity check (CPU↔MPS and MPS↔MPS at 50 steps) found loss deltas inside the 1e-3 tolerance, but CPU remains the anchor regardless; stated honestly in README |
| Time slips (job applications, other modules) | Cut eval granularity and polish items - never the Transformer itself; parking lot absorbs scope creep |
| Interim (Week 13) collides with exams (07-18 Dec) + experiment tail | Interim is formative (0%) so its floor is a first draft, not polish; lit synthesis completes mid-Nov (themes + gap) and drafting front-loads to ≈ 04 Dec (Fri; draft v1 milestone), so 06-18 Dec stays buffer only |
| Demo (Week 25) sits in exams (26 Apr-21 May) | Demo is a 5-min reproducible one-command run + Q&A on already-submitted claims; prep time-boxed to ~2 weeks (19-30 Apr), no new claims |

## Timeline (Gantt)

Dates below are my working estimates; module-week numbers are authoritative. W24/W25 labels are nominal module weeks (which skip vacation weeks - continuous count would run ~W28/W29); the calendar deadlines Fri 02 Apr and Fri 30 Apr govern. Supervisor meetings run fortnightly (Wed 13:30) in the background - charted here are the project phases and module deliverables only.

```mermaid
%%{init: {"gantt": {"useWidth": 910, "leftPadding": 180, "sectionFontSize": 12}}}%%
gantt
    dateFormat YYYY-MM-DD
    axisFormat %d %b
    tickInterval 2week
    todayMarker on

    section Scope (W1-2)
    Scope, canvas and project setup             :done, plan, 2026-09-24, 2026-09-30

    section Literature (W1-13)
    Ongoing reading + annotation (zotero) :active, lit, 2026-09-24, 2026-12-19
    Literature synthesis (themes + gap)                :synth, 2026-10-26, 2026-11-14

    section Design (W2-4)
    Build Transformer + pipeline + tests :done, proto, 2026-10-01, 2026-10-09
    Harden proto + methods freeze (seed protocol) :done, harden, 2026-10-09, 2026-10-23

    section Experiments (W4-11)
    Sweeps (capacity / size / difficulty x 10 seeds) :exps, 2026-10-23, 2026-11-14
    Iteration buffer + grokking extension           :buf, 2026-11-16, 2026-12-01
    %% exps starts Fri 23 Oct (after harden ends 22 Oct), ends Fri 13 Nov; buffer starts Mon 16 Nov (weekend 14-15 Nov gap); absolute dates intentional

    section LSEP (W8-9)
    LSEP form preparation and submission        :lsep, 2026-11-09, 2026-11-21

    section Interim (W9-13)
    Interim report writing                      :interim, 2026-11-21, 2026-12-19
    %% draft by Fri 04 Dec; 06-18 Dec buffer only (exam weeks)

    section Testing (W13-19)
    Results analysis + confirmatory runs        :testing, 2026-12-19, 2027-02-13
    %% continuous 19 Dec-12 Feb, no Christmas pause (owner decision)

    section Writing (W18-24)
    Writing, appendices and portfolio assembly  :diss, 2027-01-25, 2027-04-03
    %% overlaps testing 25 Jan-12 Feb intentional (analysis as runs land)

    section Demo (W25)
    Demo prep (time-boxed, no new claims)        :demo, 2027-04-19, 2027-04-30

    section Deadlines
    LSEP deadline :crit, milestone, v9, 2026-11-20, 0d
    Interim deadline :crit, milestone, v13, 2026-12-18, 0d
    Portfolio deadline :crit, milestone, v24, 2027-04-02, 0d
    Demo deadline :crit, milestone, v25, 2027-04-30, 0d
```

Reading the chart: section headings are short labels (full phase names in the table above); four red diamonds mark the module submission deadlines (LSEP W9, interim W13, portfolio W24, demo W25). Overlaps are intentional: hardening (finished 22 Oct) carried the methods freeze - seed protocol, evidence retention, and a pipeline frozen before any sweep - so the sweep bar starts from a settled code state; sweeps overlap synthesis from 26 Oct with effort prioritised to sweeps once they start; dissertation overlaps testing 25 Jan-12 Feb (analysis as runs land); testing runs continuously 19 Dec-12 Feb with no Christmas pause. Parallel bars show what a period contains, not what to do simultaneously at full effort. The chart shows the intended path; the guaranteed floor if any thread stalls is documented in the risks table.

## How the plan covers the module's project cycle (W1 welcome slides)

| Module teaching timeline | Where it happens in this plan |
| --- | --- |
| W1 - project allocated, welcome | Done: canvas + first supervisor meeting |
| W2-3 - literature review | First-hand annotation of the 18 Zotero sources + synthesis (context / themes / closest work / gap) running through to the interim report; framework prototype build starts (early prototyping) |
| W4-6 - research methods & ethics | Method decisions settled in the experiment design spec; ethics stance agreed (no human data); hardening completed in this window (frozen pipeline, evidence retention, sweep tooling); proto-spec demo carried to meeting 3 (28 Oct) - meeting 2 (14 Oct) did not take place |
| W7-8 - academic writing | Workshops + the LOG.md habit; sustained writing starts with the interim report |
| W9 - ethics due | LSEP form, **due Fri 20 Nov**; content pre-agreed, submitted as soon as the form is available |
| W10-12 - no timetabled session | Project work: experiments wrap up (end of Nov); interim report written |
| W13 - interim report | Interim report (formative feedback, 0%), **due Fri 18 Dec** |
| W14 - user engagement can begin | Not applicable - synthetic data only, no human participants or personal data by design |
| W15 - software testing | Artefact testing + confirmatory evaluation (Jan); the artefact hardening it builds on was completed 22 Oct |
| W16-17 - writing final report | Dissertation writing window opens (late Jan) |
| W18-23 - continued writing/analysis | Analysis conclusions + dissertation drafting |
| W24 - dissertation due | Final portfolio, **due Fri 02 Apr 12:00** |
| W25 - demos | Demo + Q&A, **≈ Fri 30 Apr (exam weeks 25-28)** |

Marking process (for reference): supervisor and second marker grade the portfolio independently; the average is the grade; a moderator is appointed if the two differ by 3+ grade points or across pass/fail.
