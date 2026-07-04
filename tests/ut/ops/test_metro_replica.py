"""Unit tests for vllm_ascend.ops.fused_moe.metro_replica.

Covers:
- L1 position-modulo replica selection (pure algorithm, runnable on CPU torch)
- L3 global greedy (added in stage 2)
- phase gate is_decode_forward (needs vllm_ascend + ascend attention env)

The pure-algorithm tests load metro_replica.py by file path when the full
vllm_ascend package cannot be imported (e.g. a CPU-only host without
torch_npu), so they can run locally without NPU hardware.
"""
import importlib.util
import os
import types

import pytest
import torch

# Standard path (container / host with full vllm_ascend).
try:
    from vllm_ascend.ops.fused_moe.metro_replica import (  # noqa: F401
        METRO_L1_POSITION, METRO_L3_GLOBAL, build_replica_options,
        is_decode_forward, select_replica, select_replica_l1, select_replica_l3)
    HAS_VLLM_ASCEND = True
except Exception:
    # Local fallback: load metro_replica.py by path, bypassing the package
    # __init__ (which imports torch_npu). The module itself only needs torch.
    HAS_VLLM_ASCEND = False
    _HERE = os.path.dirname(os.path.abspath(__file__))
    _REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
    _MOD_PATH = os.path.join(_REPO, "vllm_ascend", "ops", "fused_moe",
                             "metro_replica.py")
    _spec = importlib.util.spec_from_file_location("metro_replica_local",
                                                   _MOD_PATH)
    _m = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_m)
    build_replica_options = _m.build_replica_options
    select_replica = _m.select_replica
    select_replica_l1 = _m.select_replica_l1
    select_replica_l3 = _m.select_replica_l3
    is_decode_forward = _m.is_decode_forward
    METRO_L1_POSITION = _m.METRO_L1_POSITION
    METRO_L3_GLOBAL = _m.METRO_L3_GLOBAL


# --------------------------------------------------------------------------- #
# L1: position-modulo replica selection (pure algorithm)
# --------------------------------------------------------------------------- #
def test_l1_distributes_tokens_across_replicas():
    # 3 logical experts with 1/2/3 replicas; physical ids span cards.
    options = torch.tensor([[10, -1, -1],   # expert0: 1 replica
                            [20, 21, -1],    # expert1: 2 replicas
                            [30, 31, 32]])   # expert2: 3 replicas
    counts = torch.tensor([1, 2, 3])
    # expert2 hit 4x (positions 0-3 % 3 -> 0,1,2,0), expert1 2x, expert0 1x.
    topk_ids = torch.tensor([2, 2, 2, 2, 1, 1, 0])
    result = select_replica_l1(topk_ids, options, counts)
    expected = torch.tensor([30, 31, 32, 30, 20, 21, 10])
    assert torch.equal(result, expected), f"{result.tolist()} != {expected.tolist()}"


def test_l1_single_replica_passthrough():
    # every expert has exactly 1 replica -> just options[topk_ids][0]
    options = torch.tensor([[5, -1], [6, -1], [7, -1]])
    counts = torch.tensor([1, 1, 1])
    topk_ids = torch.tensor([0, 2, 1, 2])
    result = select_replica_l1(topk_ids, options, counts)
    assert torch.equal(result, torch.tensor([5, 7, 6, 7]))


def test_l1_preserves_shape_2d():
    options = torch.tensor([[0, 1, -1], [2, 3, 4]])
    counts = torch.tensor([2, 3])
    topk_ids = torch.tensor([[0, 1], [1, 0]])  # [batch=2, top_k=2]
    result = select_replica_l1(topk_ids, options, counts)
    assert tuple(result.shape) == tuple(topk_ids.shape)


# --------------------------------------------------------------------------- #
# build_replica_options: options / counts / card_of_option tables
# --------------------------------------------------------------------------- #
def test_build_replica_options_returns_card_of_option():
    # [card=2, expert=2], valid_count=1 -> physical_id = slot + card * 1
    # expert0 on card0 (slot0 -> phys0) and card1 (slot0 -> phys1)
    # expert1 on card0 (slot0 -> phys0) only
    gem = torch.tensor([[0, 0], [0, -1]])
    options, counts, card_of_option = build_replica_options(gem, 2, 1)
    assert torch.equal(options, torch.tensor([[0, 1], [0, -1]]))
    assert torch.equal(counts, torch.tensor([2, 1]))
    # card_of_option records the REAL card of each replica, not the replica idx
    assert torch.equal(card_of_option, torch.tensor([[0, 1], [0, 0]]))


# --------------------------------------------------------------------------- #
# L3: global greedy (per-expert, Lemma 1, correct card mapping)
# --------------------------------------------------------------------------- #
def test_l3_single_replica_expert_uses_first_replica():
    # expert1 has only 1 replica (d=1) -> greedy skips it, picks replica 0
    options = torch.tensor([[0, 1], [0, -1]])
    counts = torch.tensor([2, 1])
    card_of_option = torch.tensor([[0, 1], [0, 0]])
    topk_ids = torch.tensor([0, 0, 0, 1])  # expert0 x3, expert1 x1
    result = select_replica_l3(topk_ids, options, counts, card_of_option, 2)
    # expert0 (hot) -> card0 (phys0); expert1 (d=1) -> phys0
    assert torch.equal(result, torch.tensor([0, 0, 0, 0]))


def test_l3_greedy_balances_experts_across_cards():
    # two experts, each with replicas on card0 and card1; expert0 hotter
    options = torch.tensor([[0, 1], [2, 3]])
    counts = torch.tensor([2, 2])
    card_of_option = torch.tensor([[0, 1], [0, 1]])
    topk_ids = torch.tensor([0, 0, 1])  # expert0 (2) hotter than expert1 (1)
    result = select_replica_l3(topk_ids, options, counts, card_of_option, 2)
    # expert0 -> card0 (phys0); expert1 avoids loaded card0 -> card1 (phys3)
    assert torch.equal(result, torch.tensor([0, 0, 3]))


def test_l3_uses_real_card_mapping_not_replica_index():
    # replicas on NON-contiguous cards (card1, card2) for expert0
    options = torch.tensor([[1, 2, -1], [0, -1, -1]])
    counts = torch.tensor([2, 1])
    card_of_option = torch.tensor([[1, 2, 0], [0, 0, 0]])
    topk_ids = torch.tensor([0, 0, 1])  # expert0 hotter
    result = select_replica_l3(topk_ids, options, counts, card_of_option, 3)
    # expert0 candidates are card1/card2 (NOT card0); picks card1 -> phys1
    assert torch.equal(result, torch.tensor([1, 1, 0]))


# --------------------------------------------------------------------------- #
# Phase gate: is_decode_forward (needs vllm_ascend + ascend attention)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not HAS_VLLM_ASCEND,
                    reason="needs full vllm_ascend env (vllm + ascend attention)")
def test_is_decode_forward_states(monkeypatch):
    from vllm_ascend.attention import attention_v1 as attn_v1
    from vllm_ascend.attention.attention_v1 import AscendAttentionState
    import vllm.forward_context as fc_mod

    class FakeMeta:
        def __init__(self, state):
            self.attn_state = state

    # Let isinstance(.., AscendMetadata) pass for our fake metadata object.
    monkeypatch.setattr(attn_v1, "AscendMetadata", FakeMeta)

    def _set(state):
        ctx = types.SimpleNamespace(attn_metadata={"x": FakeMeta(state)})
        monkeypatch.setattr(fc_mod, "get_forward_context", lambda: ctx)

    # Decode-only phases -> True
    _set(AscendAttentionState.DecodeOnly)
    assert is_decode_forward() is True
    _set(AscendAttentionState.SpecDecoding)
    assert is_decode_forward() is True
    # Prefill / chunked-prefill phases -> False
    for s in (AscendAttentionState.PrefillNoCache,
              AscendAttentionState.PrefillCacheHit,
              AscendAttentionState.ChunkedPrefill):
        _set(s)
        assert is_decode_forward() is False
    # No attn_metadata available -> conservatively False
    monkeypatch.setattr(fc_mod, "get_forward_context",
                        lambda: types.SimpleNamespace(attn_metadata={}))
    assert is_decode_forward() is False
    monkeypatch.setattr(fc_mod, "get_forward_context", lambda: None)
    assert is_decode_forward() is False
