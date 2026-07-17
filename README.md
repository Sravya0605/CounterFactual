# Counterfactual Malware Explanation — Prototype

This repository is a prototype scaffold implementing the design from
`counterfactual_malware_explanation_design.md`.

Quick start (create a virtualenv, install requirements, run CLI):

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python -m src.cli parse --input path/to/cape_report.json
```

Current contents:
Current contents:
- `src/ingestion/parser.py`: CAPE JSON -> event list parser (skeletal)
- `src/graph/graph_builder.py`: behavior graph builder and coalescing heuristics
- `src/counterfactual/search.py`: counterfactual search skeleton (placeholder)
- `src/cli.py`: simple CLI to parse a CAPE JSON and print graph stats
- `src/classifier/`: classifier harness and model wrappers (LightGBM + GNN stub)
- `src/utils/graph_features.py`: graph -> bag-of-API features for LightGBM baseline


Next steps: implement classifier harness GNN training, counterfactual engine, feasibility checker, and Tier-2 validation harness.
Next steps: implement classifier harness, counterfactual engine, feasibility checker, and Tier-2 validation harness.
