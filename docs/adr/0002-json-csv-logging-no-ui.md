# Boring observability: JSON/CSV logging, matplotlib, no UI

---
status: proposed

---

Experiment-tracking platforms (W&B, MLflow) and any web dashboard are out of scope. Decision: configs, seeds and results are logged as plain JSON/CSV, and all visualisation is matplotlib. Rationale: (1) every hour spent on UI is an hour not spent on experiments; (2) the demonstration rubric assesses understanding and ownership, not UI; (3) plain files are transparent - a future reader of the dissertation can open them without tooling. A dashboard may be revisited in Semester 2 only if all Tier-1 experiments are analysed.
