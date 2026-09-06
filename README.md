# Counterfactual Malware Explanation


## What the prototype does

- Parses CAPE-style sandbox reports into normalized events with process IDs, event categories, ATT&CK hints, and resource metadata.
- Builds a process-aware behavior graph with temporal and resource edges instead of a flat event stream.
- Generates candidate edits by deleting nodes/edges or applying curated substitutions.
- Validates candidates against process context, resource lifetimes, dependency closure, and temporal order.
- Scores the original graph first, then scores feasible edited graphs; only a confirmed verdict flip is a counterfactual.
- Exports structural candidate reports and classifier-confirmed explanation reports separately.

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
- edited graph summary statistics
- feasibility and search provenance

## Notes

This is still a research prototype rather than a production-grade malware analysis platform. Synthetic benign reports are not substitutes for real benign CAPE captures: they are useful for validating plumbing and training experiments, but can introduce generator-specific shortcuts. Actual executable correspondence still requires replaying a controlled program in CAPE; the synthetic edited trace is a structural representation, not proof of execution.

