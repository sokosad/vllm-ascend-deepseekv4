import pytest

from vllm_ascend.eplb.core.policy.policy_abstract import DynamicConfig
from vllm_ascend.eplb.core.policy.policy_default_eplb import DefaultEplb
from vllm_ascend.eplb.core.policy.policy_factory import PolicyFactory
from vllm_ascend.eplb.core.policy.policy_pool_eplb import PoolBalanceEplb
from vllm_ascend.eplb.core.policy.policy_random import RandomLoadBalance
from vllm_ascend.eplb.core.policy.policy_swift_balancer import SwiftBalanceEplb


@pytest.fixture
def dummy_config():
    return DynamicConfig()


@pytest.mark.parametrize("policy_type, expected_class", [
    (0, RandomLoadBalance),
    (1, DefaultEplb),
    (2, SwiftBalanceEplb),
    (4, PoolBalanceEplb),
    (999, RandomLoadBalance),
])
def test_generate_policy(policy_type, expected_class, dummy_config):
    policy_instance = PolicyFactory.generate_policy(policy_type, dummy_config)
    assert isinstance(policy_instance, expected_class)
