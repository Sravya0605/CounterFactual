# Counterfactual


## What the prototype does

- Parses CAPE-style sandbox reports into normalized events with process IDs, event categories, ATT&CK hints, and resource metadata.
- Builds a process-aware behavior graph with temporal and resource edges instead of a flat event stream.
- Generates simple counterfactual candidates by deleting or substituting suspicious nodes.
- Validates candidate edits against dependency-closure-style constraints.
- Exports a JSON explanation report that can be used as a starting point for analyst review or downstream evaluation.

## Project layout

- [src/ingestion/parser.py](src/ingestion/parser.py): CAPE JSON parser with process and ATT&CK-aware normalization.
- [src/graph/graph_builder.py](src/graph/graph_builder.py): entity-oriented graph builder with process and resource edges.
- [src/counterfactual/search.py](src/counterfactual/search.py): counterfactual search loop.
- [src/counterfactual/feasibility.py](src/counterfactual/feasibility.py): structural feasibility validation.
- [src/counterfactual/engine.py](src/counterfactual/engine.py): single-sample explanation engine.
- [src/cli.py](src/cli.py): CLI for parsing and explaining reports.
- [src/utils/graph_features.py](src/utils/graph_features.py): richer graph feature extraction for baseline models.
- [tests](tests): regression tests for parsing, graph building, and report generation.

## Setup

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Contents:
- `src/ingestion/parser.py`: CAPE JSON -> event list parser (skeletal)
- `src/graph/graph_builder.py`: behavior graph builder and coalescing heuristics
- `src/counterfactual/search.py`: counterfactual search skeleton (placeholder)
- `src/cli.py`: simple CLI to parse a CAPE JSON and print graph stats
- `src/classifier/`: classifier harness and model wrappers (LightGBM + GNN stub)
- `src/utils/graph_features.py`: graph -> bag-of-API features for LightGBM baseline

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

## Output format

The explanation engine writes a JSON object containing:

- the input path
- graph size statistics
- the counterfactual candidate that flipped the prototype's heuristic score
- edited graph summary statistics

## Notes

This is still a research prototype rather than a production-grade malware analysis platform. The current implementation focuses on a complete, test-backed workflow for ingestion, graph construction, feasibility-aware search, and report export, with clear seams for future expansion into richer GNN training, ATT&CK presentation layers, and larger-scale evaluation pipelines.

