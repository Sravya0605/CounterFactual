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



============================================================================


## 2026-08-15 — First real end-to-end counterfactual flip, on real data

Trained a deliberately simplified binary LightGBM (emotet vs. not,
num_leaves=4, max_depth=2, 15 rounds) on 72 held-out-split real
training samples, and ran CounterfactualSearch against an unseen
test sample (P(emotet)=0.6725, true label emotet). Result:

  status: completed
  candidate: delete_nodes ['n0', 'n164']
  P(emotet): 0.6725 -> 0.4987 (crosses 0.5 threshold)

This is the first time in the project the full pipeline (real CAPE
data -> graph -> feasibility-constrained search -> real trained
classifier) has produced a genuine flip end to end.

Inspected the two deleted nodes:
- n164 = CryptEncrypt. This IS the meaningful finding -- cryptencrypt
  was independently identified as the single strongest emotet
  discriminator in the earlier shortcut-learning analysis (100% present
  in emotet, 10%/0% in agenttesla/qbot). The search correctly located
  and removed the model's actual decision-driving evidence.
- n0 = HeapCreate. Behaviorally meaningless (near-universal bookkeeping
  API, not a real discriminator). Verified by testing n164 alone:
  produces the IDENTICAL flip (P=0.4987), while n0 alone does
  essentially nothing (P=0.6725, unchanged). This means the search's
  "minimal" 2-node answer is NOT actually minimal -- a true 1-node edit
  exists and the search didn't find it.

Root cause (not yet fixed): the enumerative candidate proposer tries
cheap single-node deletions in graph node order, not in an order that
prefers subsequently-discovered-cheaper candidates once a flip is
already found; n0 apparently got bundled in via a cascade/multi-node
path before n164-alone was tried alone in isolation. Needs
investigation in search.py's candidate generation before minimality
numbers can be trusted for any evaluation table.

Also note: this required deliberately handicapping the classifier
(num_leaves=4, max_depth=2) to get a non-saturated probability at all
-- the "real" LightGBM models trained earlier (99 trees, no depth
limit) produced ONLY 0.0000/1.0000 across all 90 samples, in-sample
AND held-out, because a handful of coarse features separate classes
so cleanly that the sigmoid saturates. This is consistent with, and
strengthens, the shortcut-learning finding: the model isn't learning
a smooth decision surface, it's making a near-binary lookup on 2-3
features. Any future counterfactual evaluation needs either (a) a
calibrated/regularized classifier, or (b) to report probability
saturation rate as its own metric, since "no_flip_found" against a
saturated model is not evidence the method doesn't work.


===================================================================

## 2026-08-16  — Held-out evaluation: zero flips, emotet-vs-agenttesla subset

Attempted a full held-out evaluation (constrained search + unconstrained
baseline comparison) on a size-filtered subset of the training data
(60 MB per-file memory cap -- necessary because some CAPE reports run
200-300+ MB and caused MemoryError when parsed). This cap structurally
excludes ALL qbot samples (smallest qbot file: 96.3 MB), so this run
covers emotet-vs-agenttesla only; qbot requires separate evaluation
with streaming JSON parsing (not yet built).

After filtering: 36 training samples (23 emotet / 13 agenttesla),
12 held-out test samples (7 emotet / 2 agenttesla, note: counts in
logs don't sum cleanly to 12, worth re-checking the split logic before
reusing this script). Retrained the same deliberately-regularized
LightGBM config (num_leaves=4, max_depth=2, 15 rounds) used in this
morning's successful test.

Result: completed_flips=0 out of 12. Two samples were already
not_malicious (orig_prob=0.2967, correctly skipped). The remaining ten
all saturate at an identical orig_prob=0.8306 and neither the
feasibility-constrained search nor an unconstrained baseline (checking
the first 30 proposed candidates) found any flip for any of them.

Also found and fixed a real bug during this investigation: the
unconstrained baseline loop did not check whether orig_prob was already
below threshold before searching, so two already-not_malicious samples
were being miscounted as trivial cost-1 "flips" (edits that changed
nothing, since new_prob equaled the unchanged orig_prob). Fixed by
skipping samples already below threshold, matching what the
constrained search already did correctly.

Interpretation: this is a genuine null result, not a bug artifact
(confirmed by rerunning after the fix). It shows this morning's
successful flip (P: 0.6725 -> 0.4987, logged earlier today) does not
generalize automatically -- a different training subset (36 vs 72
samples, different class balance, qbot excluded) produced a model
saturated enough that no candidate within budget moves any sample.
This is a stability/consistency question for the search+classifier
pipeline that needs its own dedicated investigation, not something to
resolve by further tweaking this evaluation script tonight.

Known open issues, not yet fixed:
- 60 MB memory cap structurally excludes qbot entirely; needs streaming
  JSON parsing to include large reports without loading the whole file
  into memory at once.
- Held-out test set counts (7 emotet + 2 agenttesla = 9, not 12) don't
  reconcile with heldout_total=12 in the summary -- needs checking
  before this script's numbers are trusted further.



=============================================================================

## 2026-08-16 (evening) — Held-out evaluation: zero flips, emotet-vs-agenttesla subset

Ran a full held-out evaluation (feasibility-constrained search +
unconstrained baseline comparison) on a size-filtered subset of the
training data. A 60 MB per-file cap was required because some CAPE
reports run 200-300+ MB on disk and cause MemoryError when parsed
(observed directly: 307.6 MB file -> MemoryError in json.load()). This
cap structurally excludes ALL qbot samples (smallest qbot file in the
corpus: 96.3 MB), so this evaluation covers emotet-vs-agenttesla only;
qbot requires a separate evaluation with streaming JSON parsing
(not yet built, tracked as follow-up work).

After filtering: 48 training rows -> 36 graphs actually loaded
(23 emotet / 13 agenttesla), 12 held-out test rows -> 9 graphs actually
evaluated (7 emotet / 2 agenttesla). Retrained the same deliberately-
regularized LightGBM config (num_leaves=4, max_depth=2, 15 rounds)
used in the earlier same-day successful flip test.

RESULT: completed_flips=0 out of 9 evaluated (feasibility_rate=0.00%).
Two samples were already not_malicious (orig_prob=0.2967, correctly
skipped by both searches). The remaining seven all saturate at an
identical orig_prob=0.8306; neither the feasibility-constrained search
nor an unconstrained baseline (first 30 proposed candidates) found any
flip for any of them.

Two real bugs found and fixed during this investigation, both
confirmed via before/after diagnostic output, not assumed:
1. Unconstrained baseline loop didn't check orig_prob against threshold
   before searching, so already-not_malicious samples were miscounted
   as trivial cost-1 "flips" (edits that changed nothing -- new_prob
   equaled the unchanged orig_prob). Fixed: skip samples already below
   threshold, matching the constrained search's existing behavior.
2. Summary's heldout_total was computed from the PRE-size-filter row
   count (12) instead of the actual number of graphs evaluated (9),
   which would have silently corrupted feasibility_rate and any other
   per-sample-averaged metric as soon as any sample got skipped for any
   reason. Fixed: heldout_total_evaluated now drives all rate
   denominators; heldout_total_before_size_filter kept as a separate,
   transparent line.

INTERPRETATION: this is a genuine, internally-consistent null result,
not a bug artifact -- confirmed by rechecking after both fixes, with
final output containing no remaining contradictions between the
per-family counts, per-sample tables, and summary totals. It shows
today's earlier successful flip (P: 0.6725 -> 0.4987, logged in the
2026-08-16 entry above) does not generalize automatically: a
differently-sized, differently-balanced training subset (36 samples,
qbot excluded, vs. the earlier 72-sample 3-family split) produced a
model saturated enough that no candidate within the search budget
moves any of its confidently-classified samples. This is a genuine
stability/consistency finding about the search+classifier pipeline
under retraining, worth its own dedicated investigation before any
feasibility-rate number from this pipeline is used in a results table.

Known open issues, not yet fixed:
- 60 MB memory cap structurally excludes qbot; needs streaming JSON
  parsing (parse incrementally, extract features, discard raw graph)
  to include large reports without multi-GB memory spikes.
- Model saturation appears sensitive to which training subset is used;
  needs systematic investigation (does this hold across multiple
  random splits, or was the earlier successful flip itself the
  outlier?) before drawing conclusions either way.




===================================================================================


## 2026-08-17 — 5-seed stability check: feasibility rate 0.00% across all seeds

Ran the full held-out evaluation across 5 independent random seeds
(0-4) on the same emotet-vs-agenttesla, 60MB-size-filtered subset used
in the earlier same-day evaluation. Result: feasibility_rate_min=0.00%,
feasibility_rate_max=0.00%, feasibility_rate_mean=0.00%, 0 completed
flips out of 44 total held-out samples evaluated across all 5 seeds
combined.

This resolves the open question from the earlier same-day entry: the
single successful flip found this morning (P: 0.6725 -> 0.4987) was
the outlier, not representative. Five independent train/test splits,
each retraining the same deliberately-regularized LightGBM config
(num_leaves=4, max_depth=2, 15 rounds), consistently found zero
feasible flips.

Likely root cause: all 5 seeds' training runs show a near-identical
positive-class base rate (pavg range 0.629-0.694) and the same
"[Warning] No further splits with positive gain" pattern repeating
dozens of times per run -- consistent with this morning's direct
observation that this model configuration collapses to only 2-3
distinct output probability values regardless of input graph. Whether
any given sample's output happens to land close enough to the 0.5
threshold for the search to cross it within budget appears to be
essentially chance, not a stable property of the method.

CONCLUSION: this specific classifier configuration (artificially
shallow, chosen to get non-saturated probabilities out of an otherwise-
saturated LightGBM) is too coarse and seed-sensitive to support
reliable counterfactual evaluation. This is a classifier-quality
problem, not a search-engine problem -- the search+feasibility
machinery itself has been separately verified correct (candidate-
starvation fix, minimality check on 1-2 op candidates, feasibility
checker's full test suite). Before any feasibility-rate number is
usable in a results table, the classifier needs either (a) more
training data per family, (b) proper regularization/calibration
instead of an artificial num_leaves=4 cap, or (c) both. This is now
the top-priority open item for the project, ahead of scaling to more
seeds/samples with the current classifier.

Note: this evaluation script (scripts/holdout_evaluation.py) was
substantially rewritten in this pass to add the 5-seed loop; the
per-sample results table, the minimality_check function, and the
unconstrained-baseline cost comparison were removed/simplified in the
rewrite and are not currently present. Those need to be restored
before the next evaluation pass, alongside the classifier fix above.