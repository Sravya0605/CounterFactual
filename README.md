# Counterfactual Malware Explanation


## What the prototype does

- Parses CAPE-style sandbox reports into normalized events with process IDs, event categories, ATT&CK hints, and resource metadata.
- Builds a process-aware behavior graph with temporal and resource edges instead of a flat event stream.
- Generates candidate edits by deleting nodes/edges or applying curated substitutions.
- Validates candidates against process context, resource lifetimes, dependency closure, and temporal order.
- Scores the original graph first, then scores feasible edited graphs; only a confirmed verdict flip is a counterfactual.
- Writes classifier-confirmed explanations; synthetic Tier-2 round-trip validation stays in memory.

## Project layout

- [src/ingestion/parser.py](src/ingestion/parser.py): CAPE JSON parser with process and ATT&CK-aware normalization.
- [src/graph/graph_builder.py](src/graph/graph_builder.py): entity-oriented graph builder with process and resource edges.
- [src/counterfactual/search.py](src/counterfactual/search.py): candidate generation, feasibility filtering, and classifier-confirmed flip search.
- [src/counterfactual/feasibility.py](src/counterfactual/feasibility.py): structural feasibility validation.
- [src/counterfactual/engine.py](src/counterfactual/engine.py): single-sample explanation engine.
- [src/cli.py](src/cli.py): CLI for parsing and explaining reports.
- [scripts/generate_benign_reports.py](scripts/generate_benign_reports.py): synthetic benign CAPE data for pipeline tests only.
- [scripts/train_malware_benign_classifier.py](scripts/train_malware_benign_classifier.py): balanced malware-vs-benign LightGBM training and holdout manifest.
- [src/utils/graph_features.py](src/utils/graph_features.py): richer graph feature extraction for baseline models.
- [tests](tests): regression tests for parsing, graph building, and report generation.

## Setup

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The classifier is an evaluation oracle, not the counterfactual generator. The lifecycle is:

1. Score the original behavior graph.
2. Propose edits and reject infeasible candidates.
3. Apply feasible candidates in increasing edit cost.
4. Score each edited graph.
5. Report only a candidate whose classifier verdict actually flips.

## Quick start

Parse the bundled sample report:

```powershell
python -m src.cli parse --input tests/sample_cape.json
```

Run the full explanation prototype on the sample:

```powershell
python -m src.cli demo --out demo_out.json
```

Run the explanation workflow against a custom CAPE report:

```powershell
python -m src.cli explain --input path/to/cape_report.json --out explain_out.json
```

Generate synthetic benign reports for plumbing checks:

```powershell
python scripts/generate_benign_reports.py --count 400 --out-dir data/benign_reports --csv-path data/benign_batch.csv
```

Train a balanced malware-vs-benign oracle using 400 retained malware reports
and 400 synthetic benign reports. The script reserves 100 malware reports as
a holdout for lifecycle testing:

```powershell
python scripts/train_malware_benign_classifier.py --malware-count 400 --benign-count 400
```

Use the resulting model explicitly with a report:

```powershell
python -m src.cli explain --input path/to/report.json --out explanation.json --backend lgbm
```

## Output format

The explanation engine writes a JSON object containing:

- the input path
- graph size statistics
- the classifier-confirmed counterfactual candidate, when a flip is found
- a readable flip summary and API-level details for each edit
- edited graph summary statistics
- feasibility and search provenance

The full pipeline writes an explanation for every successful flip into the
counterfactual output directory:

- `<sample>_counterfactual_explanation.json`: classifier scores, candidate
  edits, per-edit node/API details, and in-memory Tier-2 round-trip results

## Full current-corpus run

For the current `data/` corpus (757 benign reports and 1,049 malware reports),
this command trains on all benign reports and a seeded sample of 839 malware
reports, then searches the remaining 210 malware reports. Counterfactual
reports are written to a separate directory so earlier results are preserved:

```powershell
python run_full_pipeline.py --backend gnn --seed 0 --max-per-class 839 --max-counterfactual 210 --per-file-timeout 1800 --out pipeline_complete_dataset_results.json --counterfactual-output-dir data/counterfactual_reports_complete
```

The run writes a JSONL checkpoint beside the results file as each malware
report completes. If interrupted, rerun the same command with `--resume` to
reuse matching completed results; errored or timed-out reports are retried.

## Boilerplate-excluded counterfactual run

Compute boilerplate API labels from the exact training reports in a saved
pipeline run, then pass that JSON to the restricted candidate search:

```powershell
python scripts\analyze_training_api_prevalence.py --results pipeline_results.json --out api_prevalence.json
python run_full_pipeline.py --backend gnn --seed 7 --max-per-class 600 --max-counterfactual 300 --reuse-model --exclude-boilerplate --boilerplate-apis-file api_prevalence.json --per-file-timeout 1800 --out pipeline_results_boilerplate_excluded.json --counterfactual-output-dir data\counterfactual_reports_boilerplate_excluded
```

`--reuse-model` loads `models\pipeline_gnn.pkl` instead of retraining or
overwriting it. The exclusion is applied during candidate generation,
including single-node pools, combinations, and trio extensions. Timed-out
reports are recorded as `timed_out` in the results and checkpoint files.

### Current investigation snapshot (October 11, 2026)

The saved seed-7 baseline in `pipeline_results.json` used 600 reports per
class, a 20% held-out split, and tested 300 malware reports. Held-out accuracy
was **97.92%**. Of the 300 counterfactual reports, **25 flipped**, **268**
had no flip found, and **7** were already below the malicious threshold.
Therefore, the baseline rate is **25/293 = 8.5%** among reports initially
classified as malicious (approximate 95% interval **5.8–12.3%**). This is a
single-model, single-seed result, not a multi-seed estimate.

Among the 25 selected flips, mean `new_prob` was **0.375** and mean
`best_prob` was **0.317**; **17/25 (68%)** selected edits had `new_prob`
below **0.45**. `new_prob` is the selected explanation's score; `best_prob`
is the minimum score among the candidates evaluated.

The prevalence analysis in `api_prevalence_oct10_final.json` found 44
boilerplate candidates using the baseline's exact training split. The
offline filtered-rescore analysis retained at least one flipping saved
candidate for **7/25** baseline flips. This is not the same as rerunning
candidate search with the exclusions enabled and must not be reported as the
restricted pipeline's flip rate. Nearest-node controls on the same reports
had **440 available**, **155 scored**, and **20 flips**; see
`api_flip_dependence_oct10_final.jsonl` for per-report results and match
distances.

For matched restricted-versus-unrestricted seed comparisons, first train and
save a model with the unrestricted run for that seed. Then run the restricted
arm with `--reuse-model` before training another seed, so both arms use the
same model. Change the seed and output paths for each pair. Do not use
`--reuse-model` for the unrestricted training arm if it should initialize a
new model. Report timed-out files separately as **not completed**.

The feasibility-gate-off flip-rate comparison, equal-budget random-deletion
baseline, three-seed summary, and WinMET/unileon external test evaluation
have not been completed. The existing WinMET batch builder targets the
training-report directory; do not merge external evaluation reports into
training. Use separately staged test reports and labels for any external
evaluation. Consult [SEARCH_INVESTIGATION_SUMMARY.md](SEARCH_INVESTIGATION_SUMMARY.md)
for the detailed evidence and run status.

## Notes

This is still a research prototype rather than a production-grade malware analysis platform. Synthetic benign reports are not substitutes for real benign CAPE captures: they are useful for validating plumbing and training experiments, but can introduce generator-specific shortcuts. Actual executable correspondence still requires replaying a controlled program in CAPE; the synthetic edited trace is a structural representation, not proof of execution.
