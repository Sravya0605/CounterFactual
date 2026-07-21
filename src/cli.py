"""Command-line interface for the counterfactual malware explanation prototype."""
import argparse
import json
from pathlib import Path

from src.counterfactual.engine import CounterfactualEngine
from src.graph.graph_builder import build_behavior_graph
from src.ingestion.parser import parse_cape_json


def cmd_parse(args):
    events = parse_cape_json(args.input)
    graph = build_behavior_graph(events)
    payload = {
        "input": args.input,
        "events": len(events),
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "apis": sorted({data.get("api", "unknown") for _, data in graph.nodes(data=True)}),
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(
            f"Parsed {payload['events']} events -> graph nodes={payload['nodes']} edges={payload['edges']}"
        )


def cmd_explain(args):
    engine = CounterfactualEngine(classifier_backend=args.backend)
    result = engine.explain_and_write(args.input, args.out)
    summary = result.get("result", {})
    status = summary.get("status", "completed")
    print(f"Explanation written to {args.out}")
    print(f"Status: {status}")
    print(f"Graph nodes: {result.get('graph_nodes')} edges: {result.get('graph_edges')}")
    if summary.get("candidate"):
        print(f"Candidate: {summary['candidate']}")
    else:
        print("No counterfactual flip found for the sample.")


def cmd_demo(args):
    default_input = Path(__file__).resolve().parents[1] / "tests" / "sample_cape.json"
    cmd_explain(argparse.Namespace(input=str(default_input), out=args.out, backend=args.backend))


def build_parser():
    parser = argparse.ArgumentParser("counterfactual-proto")
    subparsers = parser.add_subparsers(dest="cmd")

    parse_parser = subparsers.add_parser("parse", help="Parse a CAPE JSON report and summarize the graph")
    parse_parser.add_argument("--input", required=True, help="Path to CAPE JSON report")
    parse_parser.add_argument("--json", action="store_true", help="Emit structured JSON output")

    explain_parser = subparsers.add_parser("explain", help="Generate a counterfactual explanation")
    explain_parser.add_argument("--input", required=True, help="Path to CAPE JSON report")
    explain_parser.add_argument("--out", default="explain_out.json", help="Path to write explanation JSON")
    explain_parser.add_argument(
        "--backend",
        default="heuristic",
        choices=["heuristic", "lgbm", "gnn", "sklearn"],
        help="Classifier backend to use",
    )

    demo_parser = subparsers.add_parser("demo", help="Run the prototype against the bundled sample report")
    demo_parser.add_argument("--out", default="demo_out.json", help="Path to write explanation JSON")
    demo_parser.add_argument(
        "--backend",
        default="heuristic",
        choices=["heuristic", "lgbm", "gnn", "sklearn"],
        help="Classifier backend to use",
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.cmd == "parse":
        cmd_parse(args)
    elif args.cmd == "explain":
        cmd_explain(args)
    elif args.cmd == "demo":
        cmd_demo(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
