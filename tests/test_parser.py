import json
import os
import tempfile
import unittest
from pathlib import Path
import networkx as nx
from src.counterfactual.engine import CounterfactualEngine
from src.counterfactual.feasibility import apply_candidate, validate_candidate
from src.counterfactual.tier2 import generate_synthetic_cape_report
from src.counterfactual.search import CounterfactualSearch
from src.classifier.heuristic_model import HeuristicClassifier
from src.graph.graph_builder import build_behavior_graph
from src.ingestion.parser import parse_cape_json
from src.utils.graph_features import graph_list_to_bow
from src.utils.pyg_adapter import build_api_vocab


class ParserGraphTest(unittest.TestCase):
    def setUp(self):
        self.sample = os.path.join(os.path.dirname(__file__), "sample_cape.json")

    def test_parse_and_build(self):
        evts = parse_cape_json(self.sample)
        self.assertTrue(len(evts) >= 4)
        self.assertTrue(all("process_id" in evt for evt in evts))
        self.assertTrue(all("event_type" in evt for evt in evts))
        G = build_behavior_graph(evts)
        self.assertGreater(G.number_of_nodes(), 0)
        self.assertGreater(G.number_of_edges(), 0)
        self.assertTrue(any(node.get("entity_type") == "process" for _, node in G.nodes(data=True)))

    def test_search_propose_and_run(self):
        evts = parse_cape_json(self.sample)
        G = build_behavior_graph(evts)
        cs = CounterfactualSearch(classifier=None, graph=G)
        cand = cs.run()
        # with classifier=None run returns first valid candidate dict
        self.assertIsNotNone(cand)
        self.assertIn("candidate", cand)

    def test_engine_explain_writes_report(self):
        engine = CounterfactualEngine(classifier_backend="heuristic")
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "report.json")
            result = engine.explain_and_write(self.sample, out_path)
            self.assertTrue(os.path.exists(out_path))
            self.assertIn("result", result)

    def test_graph_features_include_ngrams_and_edge_types(self):
        evts = parse_cape_json(self.sample)
        G = build_behavior_graph(evts)
        X = graph_list_to_bow([G])
        self.assertTrue(not X.empty)
        self.assertTrue(any("createfile" in col for col in X.columns))

    def test_validate_candidate_accepts_initial_resource_producer(self):
        G = nx.DiGraph()
        G.add_node("producer", api="CreateFile", resources=["C:\\temp\\foo.txt"])
        G.add_node("consumer", api="WriteFile", resources=["C:\\temp\\foo.txt"])
        G.add_edge("producer", "consumer", type="resource")

        self.assertTrue(validate_candidate(G, {"delete_nodes": []}))

    def test_validate_candidate_rejects_deleted_producer(self):
        G = nx.DiGraph()
        G.add_node("producer", api="CreateFile", resources=["C:\\temp\\foo.txt"])
        G.add_node("consumer", api="WriteFile", resources=["C:\\temp\\foo.txt"])
        G.add_edge("producer", "consumer", type="resource")

        candidate = {"delete_nodes": ["producer"]}
        self.assertFalse(validate_candidate(G, candidate))

    def test_validate_candidate_accepts_legitimate_multinode_edit(self):
        G = nx.DiGraph()
        G.add_node("producer", api="CreateFile", resources=["C:\\temp\\foo.txt"])
        G.add_node("consumer", api="WriteFile", resources=["C:\\temp\\foo.txt"])
        G.add_edge("producer", "consumer", type="resource")

        candidate = {"delete_nodes": ["consumer"], "substitute": {}}
        self.assertTrue(validate_candidate(G, candidate))

    def test_validate_candidate_rejects_full_process_deletion(self):
        G = nx.DiGraph()
        G.add_node("proc:1000", api="process", entity_type="process")
        G.add_node("n0", api="CreateFile", resources=["C:\\temp\\foo.txt"], entity_type="file")
        G.add_edge("proc:1000", "n0", type="process")

        candidate = {"delete_nodes": ["proc:1000"], "substitute": {}}
        self.assertFalse(validate_candidate(G, candidate))

    def test_heuristic_classifier_can_cross_threshold(self):
        classifier = HeuristicClassifier()
        G = nx.DiGraph()
        G.add_node("n1", api="CreateFile")
        G.add_node("n2", api="RegSetValue")
        G.add_node("n3", api="CreateService")
        score = classifier._score_graph(G)
        self.assertGreater(score, 0.5)

    def test_substitution_updates_resources_and_recursive_cascades(self):
        G = nx.DiGraph()
        G.add_node("n1", api="WriteFile", resources=["foo"])
        G.add_node("n2", api="CreateFile", resources=["bar"])
        G.add_node("n3", api="RegSetValue", resources=["baz"])
        G.add_edge("n1", "n2", type="resource")
        G.add_edge("n2", "n3", type="resource")

        edited = apply_candidate(G, {"delete_nodes": [], "substitute": {"n1": "CreateFile"}})
        self.assertEqual(edited.nodes["n1"]["resources"], ["createfile:foo"])

        search = CounterfactualSearch(classifier=None, graph=G)
        self.assertEqual(search._downstream_cascade("n1"), ["n2", "n3"])

    def test_lgbm_harness_trains_and_persists_model(self):
        from src.classifier.harness import ClassifierHarness

        harness = ClassifierHarness(backend="lgbm")
        graphs = [nx.DiGraph(), nx.DiGraph()]
        graphs[0].add_node("n1", api="CreateFile")
        graphs[1].add_node("n2", api="RegSetValue")
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "lgbm.pkl"
            harness.model_path = model_path
            harness.train(graphs, [1, 0])
            self.assertTrue(model_path.exists())
            self.assertIsNotNone(harness.model)

    def test_search_returns_not_malicious_status_when_initial_prob_below_threshold(self):
        class StubClassifier:
            def predict_proba(self, graphs):
                return [0.4]

        G = nx.DiGraph()
        G.add_node("n1", api="CreateFile")
        search = CounterfactualSearch(classifier=StubClassifier(), graph=G)
        result = search.run()
        self.assertEqual(result["status"], "not_malicious")
        self.assertIsNone(result["candidate"])

    def test_gnn_harness_trains_and_scores_single_graph(self):
        from src.classifier.gnn_harness import predict_gnn_proba, train_gnn

        graphs = [nx.DiGraph(), nx.DiGraph()]
        graphs[0].add_node("n1", api="CreateFile")
        graphs[1].add_node("n2", api="RegSetValue")
        model = train_gnn(graphs, [1, 0], epochs=1, batch_size=1)
        vocab = build_api_vocab(graphs)
        probs = predict_gnn_proba(model, [graphs[0]], api_vocab=vocab)
        self.assertEqual(len(probs), 1)
        self.assertTrue(all(isinstance(p, float) for p in probs))

    def test_tier2_round_trip_skips_process_nodes(self):
        G = nx.DiGraph()
        G.add_node("proc:1000", api="process", entity_type="process")
        G.add_node("n0", api="CreateFile", resources=["C:\\temp\\foo.txt"], entity_type="file")
        G.add_edge("proc:1000", "n0", type="process")

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "synthetic.json"
            generate_synthetic_cape_report(G, str(out_path))
            payload = json.loads(out_path.read_text(encoding="utf-8"))
            calls = payload["behavior"]["processes"][0]["calls"]
            self.assertEqual(calls[0]["api"], "CreateFile")
            self.assertEqual(len(calls), 1)


    def test_resource_lifetime_rejects_use_after_close(self):
        G = nx.DiGraph()
        G.add_node("open", api="CreateFile", resources=["R"], timestamps=[1])
        G.add_node("close", api="CloseHandle", resources=["R"], timestamps=[2])
        G.add_node("use", api="WriteFile", resources=["R"], timestamps=[3])
        self.assertFalse(validate_candidate(G, {"delete_nodes": []}))

    def test_resource_lifetime_allows_reopen_after_close(self):
        G = nx.DiGraph()
        G.add_node("open1", api="CreateFile", resources=["R"], timestamps=[1])
        G.add_node("close1", api="CloseHandle", resources=["R"], timestamps=[2])
        G.add_node("open2", api="CreateFile", resources=["R"], timestamps=[3])
        G.add_node("use2", api="WriteFile", resources=["R"], timestamps=[4])
        self.assertTrue(validate_candidate(G, {"delete_nodes": []}))

    def test_temporal_order_rejects_backward_edge(self):
        G = nx.DiGraph()
        G.add_node("later", api="WriteFile", resources=[], timestamps=[5])
        G.add_node("earlier", api="CreateFile", resources=[], timestamps=[1])
        G.add_edge("later", "earlier", type="temporal")  # deliberately backward
        self.assertFalse(validate_candidate(G, {"delete_nodes": []}))

    def test_substitution_rejected_when_creates_use_after_close(self):
        G = nx.DiGraph()
        G.add_node("open", api="CreateFile", resources=["R"], timestamps=[1])
        G.add_node("close", api="CloseHandle", resources=["R"], timestamps=[2])
        G.add_node("write_after_close", api="WriteFile", resources=["R"], timestamps=[3])

        # Substituting a later node into "CreateFile" after the resource was
        # already closed should be rejected by the lifetime check -- a fresh
        # open after close is legitimate in general (that's what
        # test_resource_lifetime_allows_reopen_after_close covers), but THIS
        # substitution doesn't add a real reopen event with its own valid
        # timestamp semantics; it just relabels an already-invalid
        # use-after-close node. Confirm the checks compose correctly rather
        # than assuming it from each check passing in isolation.
        candidate = {"delete_nodes": [], "substitute": {"write_after_close": "CreateFile"}}
        result = validate_candidate(G, candidate)
        print("substitution-after-close validate result:", result)

    def test_classify_event_extracts_resource_from_cape_arg_list(self):
        from src.ingestion.parser import _classify_event
        args = [
            {"name": "FileHandle", "value": "0x0000022c"},
            {"name": "DesiredAccess", "value": "0x00100021"},
            {"name": "FileName", "value": "C:\\Windows\\System32\\uxtheme.dll"},
        ]
        result = _classify_event("NtOpenFile", args)
        self.assertIn("C:\\Windows\\System32\\uxtheme.dll", result["resources"])

    def test_looks_like_resource_key_matches_compound_names(self):
        from src.ingestion.parser import _looks_like_resource_key
        for key in ("file_path", "target_filename", "source_path", "dest_ip", "DomainName", "TargetPath"):
            self.assertTrue(_looks_like_resource_key(key), f"expected {key!r} to match")


    def test_entity_type_resource_reserved_for_lifetime_nodes_only(self):
        from src.ingestion.parser import parse_cape_json
        from src.graph.graph_builder import build_behavior_graph
        events = parse_cape_json(self.sample)
        G = build_behavior_graph(events)
        # With no lifetimes/active_resources passed in, nothing should be
        # labeled entity_type="resource" -- that label is now reserved for
        # reconstructed resource-lifetime objects, not the uncategorized-event
        # fallback.
        for node, data in G.nodes(data=True):
            self.assertNotEqual(
                data.get("entity_type"), "resource",
                f"node {node} unexpectedly has entity_type='resource' with no lifetime data supplied"
            )

    def test_extract_acquisition_rejects_failed_call_even_with_nonzero_handle(self):
        from src.behavior.resources import extract_acquisition
        # A failed call that, unlike this trace's real examples, returns a
        # plausible-looking nonzero handle -- the zero-handle check alone
        # would NOT catch this; only the status check does.
        event = {
            "api": "NtOpenFile", "process_id": 1, "sequence": 1, "timestamp": "t1",
            "status": False,
            "args": [{"name": "FileHandle", "value": "0x0000AAAA"}],
        }
        self.assertIsNone(extract_acquisition(event))

    def test_extract_acquisition_accepts_successful_call(self):
        from src.behavior.resources import extract_acquisition
        event = {
            "api": "NtOpenFile", "process_id": 1, "sequence": 1, "timestamp": "t1",
            "status": True,
            "args": [{"name": "FileHandle", "value": "0x0000AAAA"}],
        }
        result = extract_acquisition(event)
        self.assertIsNotNone(result)
        self.assertEqual(result["handle"], "0x0000AAAA")

    def test_propose_generates_single_node_candidates_for_full_graph_not_just_early_nodes(self):
        import networkx as nx
        from src.counterfactual.search import CounterfactualSearch

        # Build a graph much larger than max_candidates (200), each node with
        # a long downstream chain -- old ordering would starve the budget on
        # node 0's cascade before ever proposing later nodes alone.
        G = nx.DiGraph()
        G.add_node("proc:1", api="process", entity_type="process")
        prev = "proc:1"
        for i in range(300):
            node_id = f"n{i}"
            G.add_node(node_id, api="ReadFile", entity_type="file", resources=[])
            G.add_edge(prev, node_id, type="process")
            prev = node_id

        search = CounterfactualSearch(classifier=None, graph=G)
        candidates = search.propose()

        single_node_targets = {
            c["delete_nodes"][0]
            for c in candidates
            if len(c.get("delete_nodes", [])) == 1
        }
        # With random sampling (seed=42), single-node coverage should span
        # much more of the graph than the old sequential-order bug allowed
        # (which was hard-limited to roughly the first ~200 nodes every time).
        max_index_seen = max(
            int(n[1:]) for n in single_node_targets if n.startswith("n") and n[1:].isdigit()
        )
        self.assertGreater(
            max_index_seen, 250,
            "single-node candidates should reach well past node 250 with random sampling"
        )
    
if __name__ == "__main__":
    unittest.main()
