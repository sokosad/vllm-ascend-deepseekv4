import unittest

import torch

from vllm_ascend.ops.fused_moe.experts_selector import (
    _TID2EID_CACHE,
    _prepare_hash_input_ids,
    _prepare_hash_tid2eid,
)


class TestExpertsSelectorHashInputs(unittest.TestCase):

    def setUp(self):
        _TID2EID_CACHE.clear()

    def test_prepare_hash_input_ids_matches_router_tokens(self):
        input_ids = torch.tensor([7, -1, 9, 10, 11], dtype=torch.int64)

        clipped = _prepare_hash_input_ids(input_ids, 3)
        padded = _prepare_hash_input_ids(input_ids[:2], 4)

        self.assertTrue(torch.equal(clipped, torch.tensor([7, 0, 9], dtype=torch.int64)))
        self.assertTrue(torch.equal(padded, torch.tensor([7, 0, 0, 0], dtype=torch.int64)))

    def test_prepare_hash_tid2eid_clamps_invalid_dummy_values_once(self):
        tid2eid = torch.tensor([[0, 1, 999], [-3, 2, 4]], dtype=torch.int64)

        sanitized = _prepare_hash_tid2eid(tid2eid, expert_count=5)
        cached = _prepare_hash_tid2eid(tid2eid, expert_count=5)

        self.assertTrue(torch.equal(sanitized, torch.tensor([[0, 1, 4], [0, 2, 4]], dtype=torch.int32)))
        self.assertIs(cached, sanitized)


if __name__ == "__main__":
    unittest.main()
