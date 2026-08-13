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


if __name__ == "__main__":
    unittest.main()
