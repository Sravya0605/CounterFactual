"""Simple CLI to exercise the ingestion and graph builder modules."""
import argparse
from src.ingestion.parser import parse_cape_json
from src.graph.graph_builder import build_behavior_graph


def cmd_parse(args):
    evts = parse_cape_json(args.input)
    G = build_behavior_graph(evts)
    print(f"Parsed {len(evts)} events -> graph nodes={G.number_of_nodes()} edges={G.number_of_edges()}")


def main():
    p = argparse.ArgumentParser("counterfactual-proto")
    sub = p.add_subparsers(dest="cmd")
    ps = sub.add_parser("parse")
    ps.add_argument("--input", required=True, help="Path to CAPE JSON report")
        pe = sub.add_parser("explain")
        pe.add_argument("--input", required=True, help="Path to CAPE JSON report")
        pe.add_argument("--out", required=False, help="Path to write explanation JSON", default="explain_out.json")
    args = p.parse_args()
    if args.cmd == "parse":
        cmd_parse(args)
        elif args.cmd == "explain":
            from src.counterfactual.engine import CounterfactualEngine
            eng = CounterfactualEngine()
            eng.explain_and_write(args.input, args.out)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
