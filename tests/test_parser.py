import os
import tempfile
import unittest
from src.counterfactual.engine import CounterfactualEngine
from src.counterfactual.search import CounterfactualSearch
from src.graph.graph_builder import build_behavior_graph
from src.ingestion.parser import parse_cape_json
from src.utils.graph_features import graph_list_to_bow


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


if __name__ == "__main__":
    unittest.main()
