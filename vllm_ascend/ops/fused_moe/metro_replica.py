# Copyright (c) 2024 Contributors. All rights reserved.
"""Metro-inspired replica selection for MoE decode.

In decode (memory-bound), the per-step latency is dominated by the number of
unique expert weights loaded per card, not the token count (arXiv:2512.09277).
The default ep_rank % num_replicas polling assigns ALL tokens from a rank to
the SAME replica, causing load imbalance. This module distributes tokens
across replicas to minimize the number of activated experts per card.

Strategies (VLLM_ASCEND_METRO_REPLICA):
    0 = disabled (default)
    1 = L1 position-modulo (fully vectorized, no communication)
    2 = L2 greedy activation (local; card-map TODO, not yet vectorized)
    3 = L3 global greedy (full Metro: per-expert greedy + all-reduce of
        per-expert token counts across EP ranks)

Metro is decode-only: callers must gate it with is_decode_forward() so prefill
(compute-bound) keeps the default token-balanced routing.
"""
import os

import torch

# Strategy constants
METRO_DISABLED = 0
METRO_L1_POSITION = 1
METRO_L2_GREEDY = 2
METRO_L3_GLOBAL = 3


def get_metro_strategy() -> int:
    """Get the configured Metro replica selection strategy."""
    return int(os.environ.get("VLLM_ASCEND_METRO_REPLICA", "0"))


def is_decode_forward() -> bool:
    """Phase gate for Metro replica selection.

    Metro (arXiv:2512.09277) only applies to the decode (memory-bound) phase;
    prefill is compute-bound where the default token-balanced routing is
    correct. Returns True only for DecodeOnly and SpecDecoding forward passes
    (SpecDecoding is also token-dimension decode). Returns False for
    prefill/chunked-prefill, and conservatively when attn_metadata is
    unavailable so Metro never runs outside a real decode forward.

    Imports are lazy to avoid a circular dependency with the attention module
    at import time.
    """
    from vllm.forward_context import get_forward_context
    from vllm_ascend.attention.attention_v1 import (
        AscendAttentionState, AscendMetadata)

    forward_context = get_forward_context()
    if forward_context is None:
        return False
    attn_metadata = forward_context.attn_metadata
    if not attn_metadata:
        return False
    attn_meta = next(iter(attn_metadata.values()), None)
    if attn_meta is None or not isinstance(attn_meta, AscendMetadata):
        return False
    return attn_meta.attn_state in (
        AscendAttentionState.DecodeOnly, AscendAttentionState.SpecDecoding)


def build_replica_options(global_expert_map, ep_size, valid_count):
    """Build replica lookup tables from the global expert map.

    For each logical expert, list all physical expert IDs (and the card each
    lives on) across all cards. Padded with -1 (options) / 0 (card_of_option)
    for uniform tensor shape; the active prefix [:counts[expert]] is what
    matters at runtime.

    Args:
        global_expert_map: [ep_size, n_logical] tensor, slot index per card per
            expert (-1 if the expert is not on that card).
        ep_size: Number of EP ranks (cards).
        valid_count: Experts per card (physical ID = slot + card * valid_count).

    Returns:
        options: [n_logical, max_replicas] physical expert IDs (-1 padding).
        counts: [n_logical] replica count per logical expert.
        card_of_option: [n_logical, max_replicas] card id of each replica
            (0 padding). Precomputed here so the greedy router maps replicas to
            cards correctly (physical_id // valid_count), instead of the wrong
            ``replica_index % ep_size`` proxy.
    """
    n_logical = global_expert_map.shape[1]
    max_replicas = 0
    expert_to_physicals = {}
    expert_to_cards = {}
    for expert in range(n_logical):
        physicals = []
        cards = []
        for card in range(ep_size):
            slot = global_expert_map[card][expert].item()
            if slot != -1:
                physical_id = slot + card * valid_count
                physicals.append(physical_id)
                cards.append(card)
        expert_to_physicals[expert] = physicals
        expert_to_cards[expert] = cards
        if len(physicals) > max_replicas:
            max_replicas = len(physicals)

    if max_replicas == 0:
        max_replicas = 1  # avoid empty tensor

    options = torch.full((n_logical, max_replicas), -1, dtype=torch.int64)
    card_of_option = torch.zeros((n_logical, max_replicas), dtype=torch.int64)
    counts = torch.zeros(n_logical, dtype=torch.int64)
    for expert in expert_to_physicals:
        physicals = expert_to_physicals[expert]
        cards = expert_to_cards[expert]
        d = len(physicals)
        counts[expert] = d
        for i in range(d):
            options[expert][i] = physicals[i]
            card_of_option[expert][i] = cards[i]
    return options, counts, card_of_option


def select_replica_l1(topk_ids, options, counts):
    """L1: Position-modulo replica selection.

    Distributes tokens across replicas using token position modulo replica
    count. Fully vectorized, O(n) time complexity, no inter-rank communication.
    """
    num_tokens = topk_ids.shape[0]
    device = topk_ids.device

    token_counts = counts[topk_ids]  # [num_tokens]
    token_counts = token_counts.clamp(min=1)  # avoid div-by-zero

    pos_indices = torch.arange(num_tokens, device=device, dtype=torch.int64)
    replica_idx = pos_indices % token_counts  # [num_tokens]

    token_options = options[topk_ids]  # [num_tokens, max_replicas]
    selected = token_options.gather(1, replica_idx.unsqueeze(1)).squeeze(1)

    return selected.to(topk_ids.dtype)


def select_replica_l2(topk_ids, options, counts, ep_size):
    """L2: Greedy activation-minimizing replica selection (local).

    TODO: this still uses the per-token Python loop and the wrong
    ``replica_index % ep_size`` card proxy (see git history). It is kept for
    ablation but is NOT vectorized and does NOT minimize activated experts
    correctly. Prefer L3. Not wired through card_of_option yet.
    """
    num_tokens = topk_ids.shape[0]
    device = topk_ids.device

    card_load = torch.zeros(ep_size, dtype=torch.int64, device=device)
    result = torch.empty_like(topk_ids)

    token_options = options[topk_ids]
    token_counts = counts[topk_ids]

    sorted_idx = torch.argsort(topk_ids)

    for idx in sorted_idx:
        expert = topk_ids[idx].item()
        d = token_counts[idx].item()
        if d <= 1:
            result[idx] = token_options[idx][0]
            continue

        best_replica = 0
        best_load = float('inf')
        for r in range(d):
            load = card_load[r % ep_size].item() if r < ep_size else 0
            if load < best_load:
                best_load = load
                best_replica = r

        result[idx] = token_options[idx][best_replica]
        card_load[best_replica % ep_size] += 1

    return result


def select_replica_l3(topk_ids, options, counts, card_of_option, ep_size,
                      comm_group=None):
    """L3: Global greedy replica selection (full Metro), per-expert vectorized.

    Implements Algorithm 1 of arXiv:2512.09277 with Lemma 1's simplification:
    each active expert's tokens go to ONE replica (chosen on the least-loaded
    card), rather than splitting tokens across replicas.

    Pipeline:
      1. local per-expert token counts -> global via all_reduce(SUM) over EP
         ranks (equivalent to the paper's all-gather of T[1..N], cheaper).
      2. active experts sorted by global hotness (hottest first).
      3. greedy: per expert, pick the replica on the min-load card; charge that
         card with the expert's global token count.
      4. scatter the per-expert choice back to tokens (vectorized gather).

    The Python loop is over *active experts* (<= n_logical, e.g. 256), NOT over
    tokens. Per AGENTS.md, the remaining per-expert ``.item()`` calls (one
    argmin per active expert) are unavoidable control-flow syncs and are batched
    to a single host transfer of the counts/active-expert lists; there is no
    per-token synchronization.
    """
    n_logical = counts.shape[0]
    device = topk_ids.device

    # Step 1: global per-expert token counts.
    local_counts = torch.bincount(topk_ids.reshape(-1),
                                  minlength=n_logical).to(torch.int64)
    if comm_group is not None and torch.distributed.is_initialized():
        global_counts = local_counts.clone()
        torch.distributed.all_reduce(
            global_counts, op=torch.distributed.ReduceOp.SUM, group=comm_group)
    else:
        global_counts = local_counts

    # Step 2: active experts, hottest first.
    active_mask = global_counts > 0
    active_experts = active_mask.nonzero(as_tuple=False).squeeze(-1)
    flat_ids = topk_ids.reshape(-1).long()
    if active_experts.numel() == 0:
        return torch.full_like(flat_ids, options.flatten()[0].item())

    # stable=True so every EP rank produces identical expert ordering for the
    # same global_counts (tie-break by id) -> deterministic replica choices,
    # a precondition for MC2 all-to-all send/recv to match across ranks.
    order = torch.argsort(global_counts[active_experts], descending=True,
                          stable=True)
    active_experts = active_experts[order]

    # Step 3: greedy per-expert assignment (loop bounded by n_logical).
    counts_dev = counts.to(device)
    card_dev = card_of_option.to(device)
    options_dev = options.to(device)
    gcounts = global_counts.to(device)
    card_load = torch.zeros(ep_size, dtype=torch.int64, device=device)
    chosen_col = torch.zeros(n_logical, dtype=torch.int64, device=device)

    active_list = active_experts.tolist()   # 1 host transfer
    counts_list = counts_dev.tolist()        # 1 host transfer
    for expert in active_list:
        d = counts_list[expert]
        if d <= 1:
            continue  # chosen_col[expert] already 0 -> first replica
        cards_e = card_dev[expert][:d]       # [d] card ids of this expert's replicas
        loads = card_load[cards_e]           # gather loads on candidate cards
        best = int(loads.argmin().item())    # per-expert control-flow sync (<=n_logical)
        chosen_col[expert] = best
        card_load[cards_e[best]] += gcounts[expert]

    # Step 4: scatter per-expert choice to tokens (L1-style gather).
    token_col = chosen_col[flat_ids]                  # [num_tokens]
    token_options = options_dev[flat_ids]             # [num_tokens, max_replicas]
    selected = token_options.gather(1, token_col.unsqueeze(1)).squeeze(1)
    return selected.to(topk_ids.dtype)


def select_replica(topk_ids, options, counts, ep_size, strategy,
                   card_of_option=None, comm_group=None):
    """Dispatch to the configured replica selection strategy.

    Args:
        topk_ids: [num_tokens] or [batch, top_k] logical expert IDs.
        options: [n_logical, max_replicas] physical expert IDs per logical.
        counts: [n_logical] replica count per logical expert.
        ep_size: Number of EP ranks.
        strategy: METRO_L1_POSITION / METRO_L2_GREEDY / METRO_L3_GLOBAL.
        card_of_option: [n_logical, max_replicas] card id per replica (L3).
        comm_group: Optional torch.distributed group for L3 all-reduce.

    Returns:
        Physical expert IDs, same shape as topk_ids.
    """
    original_shape = topk_ids.shape
    flat_ids = topk_ids.reshape(-1).long()

    if strategy == METRO_L1_POSITION:
        result = select_replica_l1(flat_ids, options, counts)
    elif strategy == METRO_L2_GREEDY:
        result = select_replica_l2(flat_ids, options, counts, ep_size)
    elif strategy == METRO_L3_GLOBAL:
        result = select_replica_l3(flat_ids, options, counts, card_of_option,
                                   ep_size, comm_group)
    else:
        result = select_replica_l1(flat_ids, options, counts)

    return result.reshape(original_shape).to(torch.int32)
