import unittest

import networkx as nx

from src.counterfactual.search import CounterfactualSearch


class CounterfactualSearchExclusionTest(unittest.TestCase):
    def setUp(self):
        self.graph = nx.DiGraph()
        apis = [
            "NtTestAlert",
            "CreateFileW",
            "ReadFile",
            "WriteFile",
            "CloseHandle",
        ]
        for index, api in enumerate(apis):
            self.graph.add_node(
                f"n{index}",
                api=api,
                entity_type="file",
                process_id=123,
                sequences=[index],
                resources=[],
                count=1,
            )
            if index:
                self.graph.add_edge(
                    f"n{index - 1}", f"n{index}", type="temporal"
                )

    def test_excluded_api_is_removed_from_all_proposed_edit_combinations(self):
        baseline = CounterfactualSearch(self.graph).propose()
        restricted = CounterfactualSearch(
            self.graph,
            excluded_apis=["nttestalert"],
        )
        candidates = restricted.propose()

        self.assertIn(
            {"delete_nodes": ["n0"], "substitute": {}},
            baseline,
        )
        self.assertNotIn("n0", restricted._candidate_nodes())
        self.assertTrue(
            any(len(candidate.get("delete_nodes", [])) >= 3 for candidate in baseline)
        )
        self.assertTrue(candidates)
        self.assertTrue(
            all(
                not restricted._candidate_touches_excluded_api(candidate)
                for candidate in candidates
            )
        )


if __name__ == "__main__":
    unittest.main()
