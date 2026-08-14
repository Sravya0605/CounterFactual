## 2026-08-13 — Feasibility checker: process-deletion loophole

**Bug**: `validate_candidate`'s process-parent check was gated behind
`if process_nodes:` on the edited graph, so deleting every process node
silently disabled the check. The single most degenerate edit possible
(delete the process entirely) passed feasibility validation and, at
edit-cost 1, tended to dominate the minimality ranking — meaning the
system's "minimal feasible counterfactual" was structurally biased
toward a meaningless answer.

**First fix attempt (rejected)**: unconditional check — reject any
edited graph with zero process nodes. Broke 2 existing tests, because
some fixtures (isolated resource-dependency test graphs) never modeled
a process node to begin with; the check punished graphs that never
claimed to have process context, not just graphs that lost it.

**Correct fix**: transition check — reject only when the ORIGINAL graph
had a process node and the EDITED graph does not. Regression test added:
`test_validate_candidate_rejects_full_process_deletion`.

**Why this matters for the paper**: this was found by running the search
against the one real sample in the repo, not by code review alone —
supports writing this up as a concrete methodology point (test-driven
falsification of the feasibility gate) if useful for the eval section.

==============================================================================

## 2026-08-13 — Real CAPE ingestion validation

Input:
- Real CAPEv2 report: data/real_cape_report.json
- Report size: 7,313,840 bytes

Parser:
- 4,086 behavioral events extracted successfully.

Graph:
- 71 nodes
- 139 edges

Result:
- Real CAPEv2 report successfully passes through ingestion and graph construction.
- Initial failure was caused by graph_builder.py assuming numeric timestamps.
- CAPEv2 timestamps are datetime strings in the format:
  YYYY-MM-DD HH:MM:SS,mmm
- graph_builder.py was updated to normalize these timestamps before chronological ordering.

Next investigation:
- Quantify information loss caused by event coalescing:
  4,086 events → 71 graph nodes.
- Determine whether coalescing preserves the information required for dependency,
  resource-lifetime, and event-ordering feasibility constraints.



===================================================================

## 2026-08-14 — First real family classifier: strong accuracy, shallow signal

Trained LightGBM on 90 real CAPE-derived behavior graphs (30 each:
emotet, agenttesla, qbot), 6687 bag-of-features (API counts, n-grams,
edge types, entropy). Result: 5-fold CV accuracy 1.000 +/- 0.000.

Investigated because zero-variance perfect accuracy on real-world
malware data is itself a red flag, not a clean win (per the
shortcut-learning literature cited in our own survey, Section III-F).

Findings:
- Only 6-8 of 6687 features have nonzero gain in every run.
- Per-class presence-rate check confirms near-perfect single/few-feature
  separability: createprocessw (qbot 100% vs emotet 30%/agenttesla 40%),
  cryptencrypt (emotet 100% vs agenttesla 10%/qbot 0%), connectex
  (emotet/qbot 100% vs agenttesla 46.7%).
- Size-normalization control ruled out a pure graph-size shortcut
  (accuracy unchanged after normalizing by trace length).

Conclusion: the classifier is not degenerate (real trees, real splits,
real held-out generalization on this data) but is currently a coarse
API-fingerprint classifier, not evidence of rich behavioral
understanding. 90 samples across 3 families is too small/homogeneous
to distinguish "genuine family signature" from "artifact of this
sample batch." This IS the shortcut-learning finding our survey
motivates auditing for -- worth keeping and reporting honestly rather
than discarding once a bigger dataset resolves it either way.

Next: scale sample count per family substantially before trusting this
classifier as ground truth for counterfactual search; treat any
counterfactual explanation from this model with the explicit caveat
that it may be explaining an API-fingerprint shortcut, not malware
behavior, until this is revisited on a larger corpus.