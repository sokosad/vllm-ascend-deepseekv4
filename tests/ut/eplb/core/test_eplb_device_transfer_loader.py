from typing import Any
import unittest
from unittest.mock import MagicMock, patch

import torch

import vllm_ascend.eplb.core.eplb_device_transfer_loader as loader


def make_mock_adaptor():
    adaptor = MagicMock()

    adaptor.expert_map_per_layer_cpu = {
        0: {
            10: torch.tensor(1),
            20: torch.tensor(0)
        }
    }

    adaptor.expert_param_per_layer = {
        0: {
            0: [torch.tensor([1.0])],
            1: [torch.tensor([2.0])]
        }
    }

    adaptor.buffer_tensor_list = [[torch.tensor([3.0]),
                                   torch.tensor([4.0])]]
    return adaptor


def test_generate_task_and_state_flow():
    mock_adaptor = make_mock_adaptor()
    with patch("vllm_ascend.eplb.core.eplb_device_transfer_loader.get_dynamic_eplb_group", return_value=None):
        loader_obj = loader.D2DExpertWeightLoader()
    loader_obj.set_adator(mock_adaptor)

    with patch("torch.distributed.P2POp") as mock_p2p, \
         patch("torch.distributed.isend", return_value="isend_op"), \
         patch("torch.distributed.irecv", return_value="irecv_op"):

        mock_p2p.side_effect = lambda op, tensor, rank: (op, tensor, rank)

        loader_obj.state = loader.ExpertWeightUpdateState.READY
        loader_obj.generate_expert_d2d_transfer_task([(1, 10)], [(2, 20)],
                                                     {20: torch.tensor(0)}, 0)
        assert loader_obj.comm_op_list is None
        loader_obj.state = loader.ExpertWeightUpdateState.WAITING

        loader_obj.generate_expert_d2d_transfer_task([], [], {}, 0)
        assert not loader_obj.comm_op_list
        assert loader_obj.state == loader.ExpertWeightUpdateState.READY


def test_asyn_transfer_and_update():
    mock_adaptor = make_mock_adaptor()
    with patch("vllm_ascend.eplb.core.eplb_device_transfer_loader.get_dynamic_eplb_group", return_value=None):
        loader_obj = loader.D2DExpertWeightLoader()
    loader_obj.set_adator(mock_adaptor)

    loader_obj.comm_op_list = ["fake_op"]
    loader_obj.state = loader.ExpertWeightUpdateState.READY

    reqs: list[MagicMock] = []

    with patch("torch.distributed.batch_isend_irecv",
               return_value=[MagicMock(), MagicMock()]) as mock_batch, \
         patch.object(loader.D2DExpertWeightLoader, "_synchronize_device") as mock_sync:
        loader_obj.asyn_expert_weight_transfer(reqs)

    assert loader_obj.state == loader.ExpertWeightUpdateState.TRANSFERRING
    assert len(reqs) == 2
    mock_batch.assert_called_once_with(["fake_op"])
    mock_sync.assert_not_called()

    mock_req = MagicMock()
    mock_req.wait.return_value = None
    reqs = [mock_req]

    loader_obj.recv_expert_list = [(0, 0)]
    loader_obj.updated_expert_map = {20: torch.tensor(0)}
    loader_obj.updated_log2phy_map = {"dummy": 1}
    loader_obj.layer_id = 0
    loader_obj.comm_op_list = ["op"]

    with patch.object(loader.D2DExpertWeightLoader, "_synchronize_device") as mock_sync:
        loader_obj.update_expert_map_and_weight(reqs)
    mock_sync.assert_not_called()

    mock_adaptor.do_update_expert_map.assert_called_once()
    mock_adaptor.do_update_log2phy_map.assert_called_once()
    mock_adaptor.do_update_expert_weight.assert_called_once()

    assert loader_obj.state == loader.ExpertWeightUpdateState.WAITING
    assert loader_obj.recv_expert_list == []


def test_policy4_staging_uses_batched_transfer():
    fake_group = MagicMock()
    with patch("vllm_ascend.eplb.core.eplb_device_transfer_loader.get_dynamic_eplb_group",
               return_value=fake_group):
        loader_obj = loader.D2DExpertWeightLoader(policy_type=4)

    loader_obj.comm_op_list = ["fake_op"]
    loader_obj.state = loader.ExpertWeightUpdateState.READY
    reqs = []
    with patch("torch.distributed.batch_isend_irecv", return_value=[MagicMock()]) as mock_batch, \
         patch.object(loader.D2DExpertWeightLoader, "_synchronize_device") as mock_sync:
        loader_obj.asyn_expert_weight_transfer(reqs)

    mock_sync.assert_called_once()
    mock_batch.assert_called_once_with(["fake_op"])
    assert len(reqs) == 1


def test_generate_task_stages_offset_tensors():
    mock_adaptor = make_mock_adaptor()
    fake_group = MagicMock()
    fake_group.device_group = "device_group"
    with patch("vllm_ascend.eplb.core.eplb_device_transfer_loader.get_dynamic_eplb_group",
               return_value=fake_group):
        loader_obj = loader.D2DExpertWeightLoader(policy_type=4)
    loader_obj.set_adator(mock_adaptor)

    send_base = torch.arange(8.0)
    recv_base = torch.empty(8)
    send_view = send_base[2:5]
    recv_view = recv_base[3:6]
    assert send_view.storage_offset() != 0
    assert recv_view.storage_offset() != 0

    mock_adaptor.expert_map_per_layer_cpu = {0: {10: torch.tensor(0)}}
    mock_adaptor.expert_param_per_layer = {0: {0: [send_view]}}
    mock_adaptor.buffer_tensor_list = [[recv_view]]

    with patch("torch.distributed.P2POp") as mock_p2p:
        mock_p2p.side_effect = lambda op, tensor, rank, group=None: MagicMock(tensor=tensor)
        loader_obj.generate_expert_d2d_transfer_task([(1, 10)], [(2, 20)],
                                                     {20: torch.tensor(0)}, 0)

    assert loader_obj.state == loader.ExpertWeightUpdateState.READY
    send_tensor = loader_obj.comm_op_list[0].tensor
    recv_tensor = loader_obj.comm_op_list[1].tensor
    assert send_tensor.storage_offset() == 0
    assert recv_tensor.storage_offset() == 0
    assert torch.equal(send_tensor, send_view)
    assert len(loader_obj._p2p_staging_tensors) == 2
    assert loader_obj._recv_staging_tasks[0][0] is recv_view


def test_update_copies_staged_recv_buffer():
    mock_adaptor = make_mock_adaptor()
    with patch("vllm_ascend.eplb.core.eplb_device_transfer_loader.get_dynamic_eplb_group",
               return_value=None):
        loader_obj = loader.D2DExpertWeightLoader(policy_type=4)
    loader_obj.set_adator(mock_adaptor)

    dst_buffer = torch.zeros(3)
    staged_recv = torch.tensor([7.0, 8.0, 9.0])
    mock_req = MagicMock()
    mock_req.wait.return_value = None

    loader_obj.state = loader.ExpertWeightUpdateState.TRANSFERRING
    loader_obj.recv_expert_list = [(0, 0)]
    loader_obj.updated_expert_map = {20: torch.tensor(0)}
    loader_obj.updated_log2phy_map = {"dummy": 1}
    loader_obj.layer_id = 0
    loader_obj.comm_op_list = ["op"]
    loader_obj._recv_staging_tasks = [(dst_buffer, staged_recv)]
    loader_obj._p2p_staging_tensors = [staged_recv]

    with patch.object(loader.D2DExpertWeightLoader, "_synchronize_device"):
        loader_obj.update_expert_map_and_weight([mock_req])

    assert torch.equal(dst_buffer, staged_recv)
    assert loader_obj._recv_staging_tasks == []
    assert loader_obj._p2p_staging_tensors == []


def test_set_log2phy_map():
    mock_adaptor = make_mock_adaptor()
    with patch("vllm_ascend.eplb.core.eplb_device_transfer_loader.get_dynamic_eplb_group", return_value=None):
        loader_obj = loader.D2DExpertWeightLoader()
    loader_obj.set_adator(mock_adaptor)
    loader_obj.set_log2phy_map({"a": 1})
    assert loader_obj.updated_log2phy_map == {"a": 1}


def test_invalid_state_asyn_update():
    mock_adaptor = make_mock_adaptor()
    with patch("vllm_ascend.eplb.core.eplb_device_transfer_loader.get_dynamic_eplb_group", return_value=None):
        loader_obj = loader.D2DExpertWeightLoader()
    loader_obj.set_adator(mock_adaptor)

    loader_obj.state = loader.ExpertWeightUpdateState.WAITING
    reqs: list[Any] = []
    loader_obj.asyn_expert_weight_transfer(reqs)
    assert reqs == []

    loader_obj.state = loader.ExpertWeightUpdateState.READY
    loader_obj.update_expert_map_and_weight([])

    assert not mock_adaptor.do_update_expert_map.called


def load_tests(loader_obj, tests, pattern):
    suite = unittest.TestSuite()
    for test_func in (
        test_generate_task_and_state_flow,
        test_asyn_transfer_and_update,
        test_policy4_staging_uses_batched_transfer,
        test_generate_task_stages_offset_tensors,
        test_update_copies_staged_recv_buffer,
        test_set_log2phy_map,
        test_invalid_state_asyn_update,
    ):
        suite.addTest(unittest.FunctionTestCase(test_func))
    return suite
