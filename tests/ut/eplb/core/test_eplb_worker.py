import unittest

import torch

from vllm_ascend.eplb.core.eplb_worker import EplbWorker


def test_pack_update_info_returns_full_rank_plan():
    worker = EplbWorker.__new__(EplbWorker)
    worker.policy_type = 4
    worker.metro_routing = False

    send_info = {0: [(1, 2)]}
    recv_info = {1: [(0, 2)]}
    new_expert_map = torch.tensor(
        [
            [0, 1, -1],
            [0, -1, 1],
        ],
        dtype=torch.long,
    )

    packed = worker.pack_update_info([(send_info, recv_info, new_expert_map, 3)])

    assert len(packed) == 1
    record = packed[0]
    assert record["send_all"] == [[(1, 2)], []]
    assert record["recv_all"] == [[], [(0, 2)]]
    assert record["maps_all"] == [[0, 1, -1], [0, -1, 1]]
    assert record["layer_id"] == 3
    assert len(record["log2phy_all"]) == 2
    assert record["log2phy_all"][0] == record["log2phy_all"][1]


def load_tests(loader_obj, tests, pattern):
    suite = unittest.TestSuite()
    suite.addTest(unittest.FunctionTestCase(test_pack_update_info_returns_full_rank_plan))
    return suite
