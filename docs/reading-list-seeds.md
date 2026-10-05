# Reading programme - branch map + parked sources

**Zotero ("Honours Reading List") is the reference store**: all 18 sources with
metadata, PDFs and custom abstracts (claim - evidence - supports -
cautions per source, in child notes). This file only tracks
what Zotero cannot: the branch map for lit-review assembly, and the parked
sources not yet imported.

## Branch map (18 sources)

| Branch | Sources | One-line role |
| --- | --- | --- |
| Foundations / methods | Vaswani 2017; nanochat (Karpathy 2025) | Hand-written MHSA + sinusoidal PE origin; decoder-only deviation justified. nanochat = existing-system comparator (criterion 1), reference not base. nanoGPT kept as one-line deprecated predecessor only (superseded Nov 2025, repo left up for posterity). |
| Task difficulty | Power 2022; Liu 2022; Nanda 2023; Wang/Chen/Zhu 2021 | Grokking exists at our scale (Power); mechanism theory + regularisation phase effects (Liu); circuit mechanism + progress measures (Nanda); difficulty-as-factor framing - ours is task-structure variation, not curriculum (Wang). |
| Dataset size | Hestness 2017; Muennighoff 2023 | Power-law held-out error vs data size, regime established empirically (Hestness); multi-epoch regime - <=4 epochs nearly free, repetitions over parameters when data-capped (Muennighoff). |
| Capacity | Kaplan 2020; Hoffmann 2022; Levine 2020; Nakkiran 2019 | Scaling interplay, constants don't transfer (Kaplan, Hoffmann); dedicated width/depth study + grid design (Levine); double descent, operational threshold, per-seed reporting (Nakkiran). |
| Memorisation theory | Zhang 2017; Arpit 2017 | Capacity-to-memorise doesn't explain generalisation (Zhang); easy-first/noise-late dynamics, dropout slows noise fitting (Arpit). |
| Rigour / methods | Bouthillier 2021; Loshchilov 2019; Zhang 2018 | Seeds->=3 justification, per-run reporting (Bouthillier); AdamW fact + weight decay as clean variable (Loshchilov); three wd mechanisms, coefficient != strength, norm-matched control adapted (Zhang 2018). |

## Parked (do NOT import yet)

- **Olsson et al. 2022, In-context Learning and Induction Heads** + **Varma et
  al. 2023, Explaining Grokking Through Circuit Efficiency** - promote when the
  modular-arithmetic/grokking runs start (mid-Oct 2026).
- **Jiang et al., generalisation measures** - nice, not load-bearing; promote
  only if the analysis chapter needs a gap-quantification method beyond ours.
- **Alammar, The Illustrated Transformer** - background reading, not
  citation-grade.
