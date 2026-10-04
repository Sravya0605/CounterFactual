# Counterfactual Search Investigation — Current State

**Evidence snapshot:** October 3, 2026  
**Scope:** GNN-backed search, the saved `pipeline_gnn.pkl` model, and the
25-report held-out run recorded in `pipeline_results.json`.

## Executive summary

The current evidence is mixed, not an across-the-board success or failure.
In the 25-report run, 24 reports were classified malicious and **2/24
flipped**. The search substantially lowered probability for some reports,
while others barely moved. A larger graph often had a smaller observed drop
in this sample, but graph size alone does not explain the results.

The investigation found several engineering bottlenecks and search-coverage
gaps, then identified one particularly strong higher-order combination on
task576. That is evidence that combination effects can matter; it does not
show that every no-flip report contains an undiscovered interaction.

## Investigation path

### 1. Architecture hypothesis: mean/max pooling and graph size

The GNN uses mean and max graph pooling. Mean pooling can dilute a small
number of node edits on a large graph, while max pooling may depend on a
single dominant activation. This motivated checking whether large graphs
were intrinsically harder to move.

The 25-report results do not support graph size as a general explanation:
task42 (271 nodes) dropped only **0.025799**, while task584 (178 nodes)
dropped **0.398827** in that result snapshot. The largest reports, task719
(11,833 nodes) and task767 (12,105 nodes), did have two of the smallest
drops (**0.009551** and **0.013381**). This is a suggestive large-graph
pattern in a small sample, not evidence that size causes the outcome.

### 2. Performance and candidate-capacity problems

Profiling task799 (5,180 nodes, 11,612 edges) found expensive repeated
feasibility work and GNN graph conversion:

- Validation of 2,914 proposals took **724.316 s** before caching
  graph-invariant facts. A deletion-aware feasibility context and filtered
  graph view reduced that phase to **260.342 s**, with the same **926**
  feasible candidates in the compared run.
- A deletion-only GNN scorer reuses the original PyG tensors rather than
  rebuilding a complete graph representation per candidate. On 50 task799
  candidates, standard scoring took **28.638 s** and optimized scoring took
  **2.699 s**; probabilities matched exactly. On 32 task863 candidates,
  scoring was **10.972 s** versus **1.201 s**, also with exact agreement.
- The node-feature cache was active, but node-feature reuse alone did not
  remove the cost of rebuilding edge indices, edge attributes, and tensors.

Several candidate families also competed for bounded search budgets. Pair,
edge, single-deletion, closure, and triple generation have explicit caps.
Triples are direct deletions, but the 512-triple cap takes the **first 512**
combinations in enumeration order, not a probability-ranked subset.
On task42, checking the remaining **1,512** of the 2,024 top-24 triples
found 122 feasible candidates. Their best probability was **0.954173**,
compared with **0.971577** for the first 512; none flipped. The cap therefore
does affect which combinations are considered, even though the omitted
triples did not change task42's verdict.

### 3. Gradient-ranked pool wiring and task691

Gradient importance was computed for GNN candidate proposals but initially
did not determine the node order used to build pair and chain pools. Wiring
gradient importance into that ranking addressed the mismatch: task691's
previously excluded node `n247` entered the top pool.

This was a search-eligibility fix, not a guarantee of a flip. The four-node
task691 set and a similarly effective three-node set remained above the
threshold in the model snapshot used for that experiment. Subsequent model
changes produced different task691/task734 probabilities, so the historical
measurements are not interchangeable with later runs.

### 4. Flat cases versus a strong task576 interaction

For task42, task1002, and task719, the best tested top-24 direct-deletion
probabilities were:

| Report | Original | Best single | Best pair | Best of first 512 triples |
|---|---:|---:|---:|---:|
| task42 | 0.997376 | 0.987001 | 0.974433 | 0.971577 |
| task1002 | 0.998332 | 0.994455 | 0.989364 | 0.980720 |
| task719 | 0.988126 | 0.986614 | 0.985209 | 0.984084 |

The triple improvements over the best pair were **0.002856**, **0.008644**,
and **0.001125**, respectively. Exhaustively scoring the triples omitted by
the cap for task42 found a best omitted-triple probability of **0.954173**,
still above 0.5. This supports flat movement in the tested combination
spaces for these reports; it does not establish that no other edit family or
larger combination can help.

Task576 was different. On a 344-node graph, the nodes in its threshold-
crossing trio were gradient ranks **1, 2, and 6**:

- `n212` — `sysenter`
- `n206` — `NtSetInformationProcess`
- `n234` — `CryptDestroyHash`

Their individual scores were **0.922371**, **0.770308**, and **0.899974**;
the best pair scored **0.617518**; the trio scored **0.328843**. The current
search generated the trio, and both optimized and reference feasibility
checks accepted it. This is a strong observed combination effect, not proof
that gradient ranking anticipated the interaction: each node was also
individually high-ranked.

`find_flip()` originally stopped after its first threshold-crossing
candidate. It now finishes scoring the already-generated feasible list,
preserving the first flip as the selected explanation while reporting the
true minimum as `best_prob`. On task576, the selected trio's `new_prob` is
**0.328843**, while the exhaustive minimum across **416** feasible
candidates is **0.328532**.

## Held-out result snapshot

The 25 rows in `pipeline_results.json` comprise a deterministic 20-report
sample and five additional reports; they are **25 unique reports, not 45**.
The recorded run had 2 flips among 24 initially malicious reports. Mean
original probability was **0.933148**, mean reported best probability
**0.794981**, mean drop **0.138167**, and median drop **0.108976**. One
report was already below threshold.

The two recorded flips were:

| Report | Original | Reported best | Drop |
|---|---:|---:|---:|
| task228 | 0.596106 | 0.469431 | 0.126675 |
| task576 | 0.936521 | 0.328843 | 0.607679 |

The table reflects the saved result artifact from before exhaustive
post-flip scoring was added. In particular, task576's reported best is the
first crossing; its exhaustive minimum is **0.328532** as noted above.

## Timing variability

Task719 has produced identical observed search outcomes under repeated runs
with the same saved model—original probability **0.988126**, **2,797**
proposals, **646** feasible/scored candidates, and best probability
**0.978575**—but widely different elapsed times. Two consecutive same-process
runs took **270.037 s** and **298.551 s**. Other recorded runs took
**277.55 s**, **1,146.47 s**, and **3,573.17 s**.

Candidate counts and outputs did not explain the two-run difference. The
evidence is consistent with runtime/environment variation, but historical
host-load telemetry is unavailable, so its cause is unconfirmed.

## Randomness and reproducibility

Earlier repeated pipeline runs using the same ten report filenames produced
different `orig_prob` values. The confirmed code gap was that Python's random
sampling seed did not seed PyTorch's GNN weight initialization. The current
`run_full_pipeline.py` now calls `torch.manual_seed(args.seed)` for the GNN
backend, in addition to seeding Python's `random` module. Thus, GNN model
initialization and the pipeline's Python-driven sampling/splitting are seeded
for future runs with the same seed.

This improves repeatability; it does not by itself guarantee bit-for-bit
determinism across hardware, PyTorch/PyG versions, or nondeterministic
operations. Nor does it retroactively identify which seed or model generated
older result files. The `pipeline_gnn.pkl` artifact changed during this
investigation, so historical probabilities remain tied to their model
snapshot. For an auditable comparison, preserve the model artifact and record
the seed and software/runtime versions alongside each results file.

As a focused runtime check, two CPU training runs over the same three-report
fixture, with one epoch and seed 123 reset before each run, both produced
task42 probability **0.7572374939918518** (absolute difference **0.0**).
This confirms repeatability for that small fixture and environment; it is not
a repeated full-pipeline certification.

## Tier-1 search follow-up

The bounded trio-extension stage is now wired into `find_flip()`. After a
baseline no-flip result within **0.05** of the threshold, it extends up to
three best-scoring feasible deletion trios with nodes from the top-48 priority
pool. On task584 it checked **132** distinct extensions, of which **33** were
feasible, and found `{n4,n51,n68,n93}` at **0.459998608** from an original
probability of **0.900139512**. The exhaustive best scored probability was
**0.459308833**. A regression test covers the extension path and accounting.
Task799 did not meet the extension gate (baseline best probability
**0.932898163**) and completed in **126.169 s**, consistent with its prior
post-optimization timing; no extension candidates were generated.

Insertion candidates are now confirmed to be structurally feasible on all
three tested reports: task584 **45/45**, task42 **50/50**, task719 **50/50**.
Their best probabilities were **0.8595564367**, **0.9967087507**, and
**0.9875109196**, respectively, versus original probabilities
**0.9001395106**, **0.9973762035**, and **0.9881255627**. These are measurable
improvements, but none crosses the **0.5** threshold.

Substitution remains unaddressed: **6/6**, **14/14**, and **26/26** proposed
substitutions were rejected by resource-provenance feasibility checks on
task584, task42, and task719. The replacement currently changes an API node
while the node's old resource identifiers and downstream consumers remain;
the proposed operation does not establish equivalent producer/resource
semantics. Weakening the validator or fabricating replacement resources could
change acceptance correctness, so no substitution change was made.

The max-edit budget was compared at **150** and **250** through full
`find_flip()` runs on three additional near-misses:

| Report | Best at 150 | Best at 250 | Feasible/scored (150 → 250) | Proposals (150 → 250) |
|---|---:|---:|---:|---:|
| task59 | 0.527043641 | 0.527043641 | 548 → 564 | 1,673 → 1,820 |
| task299 | 0.550235987 | 0.550235987 | 753 → 753 | 2,745 → 2,745 |
| task140 | 0.684119940 | 0.684119940 | 751 → 751 | 1,878 → 1,890 |

No report's best probability or flip status changed. The higher budget did
produce 16 more scored candidates on task59 and 12 more proposals on task140,
but neither improved the minimum. This extends the earlier task299 finding:
raising the ceiling to 250 did not help these tested cases.

## Tier-2 experiments

### Full-graph random triples

To check for three-way interactions outside the gradient-ranked pool, I sampled
**1,000 unique, seeded random triples per report** on task42 and task1002.
Candidates already proposed by the current search were excluded. Each sample
was validated and feasible triples were scored by the current GNN scorer.

| Report | Original | Ranked triples feasible / best | Random triples feasible / best | Best random node gradient ranks |
|---|---:|---:|---:|---|
| task42 | 0.997376204 | 54 / 0.971576810 | 58 / 0.986980498 | 8, 71, 151 |
| task1002 | 0.998331964 | 112 / 0.980720103 | 46 / 0.994451821 | 4, 161, 417 |

The random samples' best drops from original were **0.010395706** on task42
and **0.003880143** on task1002, versus **0.025799394** and **0.017611861**
for the ranked triples. Neither random best beat the ranked result or crossed
the 0.5 threshold. The best random combinations did include low-ranked nodes,
so this is a direct negative result for these two sampled sets, not evidence
that random triples are generally unhelpful.

### Attention-pooling comparison

Added an opt-in `attention` pooling mode to `SimpleGCN`, exposed as
`--gnn-pooling attention`; the existing `mean_max` remains the default. Both
models were trained with seed **20261004**, **10 epochs**, on the same
stratified sample: **100 benign and 100 malware graphs** (each report under
5 MiB, each graph at most 5,000 nodes), with **160 training** and **40 held-out**
graphs. No model artifact was overwritten by this direct training comparison.

| Pooling | Accuracy | Macro recall | Malware recall | Benign recall | ROC-AUC | Training seconds |
|---|---:|---:|---:|---:|---:|---:|
| mean + max | 0.925 | 0.925 | 0.900 | 0.950 | 0.970 | 13.585 |
| attention | 0.800 | 0.800 | 0.850 | 0.750 | 0.850 | 9.828 |

On this small sample, attention pooling was worse by **0.125 accuracy** and
**0.120 ROC-AUC**. It is available for further experiments but is not promoted
to the default architecture. The result is preliminary and does not establish
that attention pooling is inferior on a larger, independently selected split.

### Substitution feasibility and library coverage

The earlier zero-feasible substitution result exposed two coupled issues:
common report APIs were absent from the hand-curated library, and
`apply_candidate()` replaced even same-type resource identifiers with generic
placeholders. That broke existing resource edges for Win32/NT API equivalents.
Substitution application now preserves identifiers when source and target
APIs have the same resource type. Cross-type replacements still receive a
target-typed placeholder, rooted only at the replaced event; a downstream
consumer of the old resource remains unsupported and is rejected. Tests cover
both behaviors.

The library now includes registry key open/create variants, Win32/NT process
and thread opens/creation, and file read equivalents. Substitution feasibility
on the same report samples is now:

| Report | Proposed substitutions | Feasible | Best feasible probability |
|---|---:|---:|---:|
| task584 | 75 | 74 | 0.876961350 |
| task42 | 67 | 63 | 0.996887267 |
| task719 | 1,848 | 1,843 | Not exhaustively scored |

The best feasible task584 substitution was `n81: NtMapViewOfSection ->
MapViewOfFile`; task42's was `n9: NtOpenKeyEx -> RegOpenKeyExW`. For task719,
the high feasible count demonstrates the rules operate on a large graph, but
the full candidate set was not scored because materializing and classifying
each edited 11,833-node graph is expensive. Scoring a seeded 12-candidate
sample did not complete within **600 seconds** and was stopped; task719
substitution score impact therefore remains unmeasured.

The full `find_flip()` regression check after expanding substitutions retained
the key prior outcomes: task584 still completed with the `{n4,n51,n68,n93}`
quartet (`new_prob` **0.459998608**, `best_prob` **0.459308833**), while
task42 remained `no_flip_found` with `best_prob` **0.971576810**. The expanded
search scored **449** and **456** feasible candidates respectively; neither
the substitution proposals nor the new validator path displaced those
results. All **62** directed library pairs have provenance entries.

These changes make substitution candidates available and feasibility-checked;
they do not claim every listed pair is behaviorally identical for every
trace. The library's equivalence rationale remains explicit in provenance,
and broader behavioral validation is still appropriate before interpreting
substitution wins as executable counterfactuals.

## Caveats and interpretation

- The saved GNN model changed during the investigation. A result is meaningful
  with its model snapshot; probabilities from different model states must not
  be treated as a controlled before/after comparison. PyTorch seeding is now
  present in the pipeline, but it cannot recover the initialization seed or
  model snapshot for historical results.
- `pipeline_results.json` is a generated artifact and should be read as the
  result of its recorded run, not as a timeless property of the source code.
- The triple-cap check is exhaustive only for the omitted triples of task42's
  top-24 pool. The flat-case table otherwise covers the first 512 triples.
- Feasibility checks establish structural validity under the project's
  validator, not proof that an edited trace corresponds to an executable
  malware run.
- The sample supports a mixed picture: a few strong movers and flips, several
  low-movement cases, and at least one clear three-way interaction. It is not
  enough to claim a universal size effect or to predict flippability from
  gradient ranks alone.

## Current code changes related to this investigation

- [src/counterfactual/search.py](src/counterfactual/search.py): gradient-aware
  node ranking, bounded triple proposals, bounded near-threshold trio
  extensions, exhaustive `best_prob` scoring.
- [src/counterfactual/feasibility.py](src/counterfactual/feasibility.py):
  cached graph invariants for deletion-focused validation, same-type
  substitution resource preservation, and constrained cross-type placeholders.
- [src/counterfactual/substitutions.py](src/counterfactual/substitutions.py):
  expanded common registry, file-read, process-open, and thread-creation
  substitutions with resource typing and provenance.
- [src/classifier/gnn_harness.py](src/classifier/gnn_harness.py) and
  [src/classifier/harness.py](src/classifier/harness.py): cached-base,
  deletion-only GNN scoring and an optional pooling parameter.
- [src/classifier/gnn_model.py](src/classifier/gnn_model.py) and
  [run_full_pipeline.py](run_full_pipeline.py): opt-in attention graph pooling
  via `--gnn-pooling attention`; default remains `mean_max`.
- [run_full_pipeline.py](run_full_pipeline.py): per-report `best_prob`
  output and `torch.manual_seed(args.seed)` for GNN runs.
