import unittest
from scripts.match_resource_lifetimes import match_resource_lifetimes


class MatchResourceLifetimesTest(unittest.TestCase):
    def test_interleaved_different_handles_matched_correctly(self):
        events = [
            {"id": "e1", "api": "NtOpenFile", "process_id": 1, "sequence": 1, "timestamp": "t1",
             "args": [{"name": "FileHandle", "value": "0xAAA"}]},
            {"id": "e2", "api": "NtOpenFile", "process_id": 1, "sequence": 2, "timestamp": "t2",
             "args": [{"name": "FileHandle", "value": "0xBBB"}]},
            {"id": "e3", "api": "NtClose", "process_id": 1, "sequence": 3, "timestamp": "t3",
             "args": [{"name": "Handle", "value": "0xAAA"}]},
            {"id": "e4", "api": "NtClose", "process_id": 1, "sequence": 4, "timestamp": "t4",
             "args": [{"name": "Handle", "value": "0xBBB"}]},
        ]
        result = match_resource_lifetimes(events)
        pairs = {(lt["handle"], lt["acquisition_sequence"], lt["release_sequence"]) for lt in result["lifetimes"]}
        self.assertEqual(pairs, {("0xAAA", 1, 3), ("0xBBB", 2, 4)})
        self.assertEqual(len(result["orphan_releases"]), 0)
        self.assertEqual(len(result["still_active"]), 0)

    def test_same_handle_reuse_produces_two_distinct_lifetimes(self):
        events = [
            {"id": "e1", "api": "NtOpenFile", "process_id": 1, "sequence": 1, "timestamp": "t1",
             "args": [{"name": "FileHandle", "value": "0xAAA"}]},
            {"id": "e2", "api": "NtClose", "process_id": 1, "sequence": 2, "timestamp": "t2",
             "args": [{"name": "Handle", "value": "0xAAA"}]},
            {"id": "e3", "api": "NtOpenFile", "process_id": 1, "sequence": 3, "timestamp": "t3",
             "args": [{"name": "FileHandle", "value": "0xAAA"}]},
            {"id": "e4", "api": "NtClose", "process_id": 1, "sequence": 4, "timestamp": "t4",
             "args": [{"name": "Handle", "value": "0xAAA"}]},
        ]
        result = match_resource_lifetimes(events)
        pairs = {(lt["acquisition_sequence"], lt["release_sequence"]) for lt in result["lifetimes"]}
        self.assertEqual(pairs, {(1, 2), (3, 4)})
        self.assertEqual(len(result["orphan_releases"]), 0)
        self.assertEqual(len(result["still_active"]), 0)


if __name__ == "__main__":
    unittest.main()