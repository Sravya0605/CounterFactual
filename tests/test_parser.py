import unittest
import os
from src.ingestion.parser import parse_cape_json
from src.graph.graph_builder import build_behavior_graph
from src.counterfactual.search import CounterfactualSearch


class ParserGraphTest(unittest.TestCase):
    def setUp(self):
        self.sample = os.path.join(os.path.dirname(__file__), "sample_cape.json")

    def test_parse_and_build(self):
        evts = parse_cape_json(self.sample)
        self.assertTrue(len(evts) >= 4)
        G = build_behavior_graph(evts)
        self.assertGreater(G.number_of_nodes(), 0)
        self.assertGreater(G.number_of_edges(), 0)

    def test_search_propose_and_run(self):
        evts = parse_cape_json(self.sample)
        G = build_behavior_graph(evts)
        cs = CounterfactualSearch(classifier=None, graph=G)
        cand = cs.run()
        # with classifier=None run returns first valid candidate dict
        self.assertIsNotNone(cand)
        self.assertIn("candidate", cand)


if __name__ == "__main__":
    unittest.main()
