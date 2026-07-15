#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

from unittest.mock import patch

from vllm.config import VllmConfig

from tests.ut.base import TestBase
from vllm_ascend.ascend_config import clear_ascend_config, get_ascend_config, init_ascend_config


class TestAscendConfig(TestBase):
    @staticmethod
    def _clean_up_ascend_config(func):
        def wrapper(*args, **kwargs):
            clear_ascend_config()
            func(*args, **kwargs)
            clear_ascend_config()

        return wrapper

    @_clean_up_ascend_config
    @patch("vllm_ascend.platform.NPUPlatform._fix_incompatible_config")
    def test_init_ascend_config_without_additional_config(self, mock_fix_incompatible_config):
        test_vllm_config = VllmConfig()
        # No additional config given, check the default value here.
        ascend_config = init_ascend_config(test_vllm_config)
        self.assertFalse(ascend_config.multistream_overlap_shared_expert)
        self.assertFalse(ascend_config.enable_kv_nz)

        ascend_compilation_config = ascend_config.ascend_compilation_config
        self.assertTrue(ascend_compilation_config.fuse_norm_quant)

        ascend_fusion_config = ascend_config.ascend_fusion_config
        self.assertTrue(ascend_fusion_config.fusion_ops_gmmswigluquant)

    @_clean_up_ascend_config
    @patch("vllm_ascend.platform.NPUPlatform._fix_incompatible_config")
    def test_init_ascend_config_with_additional_config(self, mock_fix_incompatible_config):
        test_vllm_config = VllmConfig()
        test_vllm_config.additional_config = {
            "ascend_compilation_config": {
                "fuse_norm_quant": False,
            },
            "ascend_fusion_config": {
                "fusion_ops_gmmswigluquant": False,
            },
            "multistream_overlap_shared_expert": True,
            "eplb_config": {"num_redundant_experts": 2},
            "refresh": True,
            "enable_kv_nz": False,
        }
        ascend_config = init_ascend_config(test_vllm_config)
        self.assertEqual(ascend_config.eplb_config.num_redundant_experts, 2)
        self.assertTrue(ascend_config.multistream_overlap_shared_expert)

        ascend_compilation_config = ascend_config.ascend_compilation_config
        self.assertFalse(ascend_compilation_config.fuse_norm_quant)
        self.assertFalse(ascend_config.enable_kv_nz)
        self.assertTrue(ascend_compilation_config.enable_npugraph_ex)
        self.assertFalse(ascend_compilation_config.enable_static_kernel)

        ascend_fusion_config = ascend_config.ascend_fusion_config
        self.assertFalse(ascend_fusion_config.fusion_ops_gmmswigluquant)

    @_clean_up_ascend_config
    @patch("vllm_ascend.platform.NPUPlatform._fix_incompatible_config")
    def test_eplb_config_with_craft_pool_layer_sizes(self, mock_fix_incompatible_config):
        test_vllm_config = VllmConfig()
        test_vllm_config.additional_config = {
            "eplb_config": {"craft_pool_layer_sizes": {"2": 1, "9": 3}},
            "refresh": True,
        }

        ascend_config = init_ascend_config(test_vllm_config)

        self.assertEqual(ascend_config.eplb_config.craft_pool_layer_sizes, {"2": 1, "9": 3})

    @_clean_up_ascend_config
    @patch("vllm_ascend.platform.NPUPlatform._fix_incompatible_config")
    def test_eplb_config_with_craft_pool_policy_knobs(self, mock_fix_incompatible_config):
        test_vllm_config = VllmConfig()
        test_vllm_config.additional_config = {
            "eplb_config": {
                "craft_pool_top_m": 16,
                "craft_pool_top_m_factor": 2,
                "craft_pool_min_hotness_delta": 0.2,
                "craft_pool_min_improvement": 0.15,
                "craft_pool_max_payback_steps": 300,
                "craft_pool_migration_cost_ratio": 2.0,
            },
            "refresh": True,
        }

        ascend_config = init_ascend_config(test_vllm_config)

        self.assertEqual(ascend_config.eplb_config.craft_pool_top_m, 16)
        self.assertEqual(ascend_config.eplb_config.craft_pool_top_m_factor, 2)
        self.assertEqual(ascend_config.eplb_config.craft_pool_min_hotness_delta, 0.2)
        self.assertEqual(ascend_config.eplb_config.craft_pool_min_improvement, 0.15)
        self.assertEqual(ascend_config.eplb_config.craft_pool_max_payback_steps, 300)
        self.assertEqual(ascend_config.eplb_config.craft_pool_migration_cost_ratio, 2.0)

    @_clean_up_ascend_config
    @patch("vllm_ascend.platform.NPUPlatform._fix_incompatible_config")
    def test_eplb_config_with_metro_routing(self, mock_fix_incompatible_config):
        test_vllm_config = VllmConfig()
        test_vllm_config.additional_config = {
            "eplb_config": {"metro_routing": True},
            "refresh": True,
        }

        ascend_config = init_ascend_config(test_vllm_config)

        self.assertTrue(ascend_config.eplb_config.metro_routing)

    @_clean_up_ascend_config
    @patch.dict(
        "os.environ",
        {
            "CRAFT_POOL_TOP_M": "12",
            "CRAFT_POOL_TOP_M_FACTOR": "3",
            "CRAFT_POOL_MIN_HOTNESS_DELTA": "0.25",
            "CRAFT_POOL_MIN_IMPROVEMENT": "0.2",
            "CRAFT_POOL_MAX_PAYBACK_STEPS": "240",
            "CRAFT_POOL_MIGRATION_COST_RATIO": "1.5",
        },
    )
    @patch("vllm_ascend.platform.NPUPlatform._fix_incompatible_config")
    def test_eplb_config_reads_craft_pool_policy_knobs_from_env(self, mock_fix_incompatible_config):
        test_vllm_config = VllmConfig()
        test_vllm_config.additional_config = {"refresh": True}

        ascend_config = init_ascend_config(test_vllm_config)

        self.assertEqual(ascend_config.eplb_config.craft_pool_top_m, 12)
        self.assertEqual(ascend_config.eplb_config.craft_pool_top_m_factor, 3)
        self.assertEqual(ascend_config.eplb_config.craft_pool_min_hotness_delta, 0.25)
        self.assertEqual(ascend_config.eplb_config.craft_pool_min_improvement, 0.2)
        self.assertEqual(ascend_config.eplb_config.craft_pool_max_payback_steps, 240)
        self.assertEqual(ascend_config.eplb_config.craft_pool_migration_cost_ratio, 1.5)

    @_clean_up_ascend_config
    @patch.dict("os.environ", {"VLLM_ASCEND_METRO_ROUTING": "true"})
    @patch("vllm_ascend.platform.NPUPlatform._fix_incompatible_config")
    def test_eplb_config_reads_metro_routing_from_env(self, mock_fix_incompatible_config):
        test_vllm_config = VllmConfig()
        test_vllm_config.additional_config = {"refresh": True}

        ascend_config = init_ascend_config(test_vllm_config)

        self.assertTrue(ascend_config.eplb_config.metro_routing)

    @_clean_up_ascend_config
    @patch("vllm_ascend.platform.NPUPlatform._fix_incompatible_config")
    def test_eplb_config_rejects_conflicting_craft_pool_layer_sizes(self, mock_fix_incompatible_config):
        test_vllm_config = VllmConfig()
        test_vllm_config.additional_config = {
            "eplb_config": {
                "craft_pool_size": 1,
                "craft_pool_layer_sizes": [0, 1],
            },
            "refresh": True,
        }

        with self.assertRaises(ValueError):
            init_ascend_config(test_vllm_config)

        clear_ascend_config()
        test_vllm_config.additional_config = {
            "eplb_config": {
                "num_redundant_experts": 16,
                "craft_pool_layer_sizes": [0, 1],
            },
            "refresh": True,
        }

        with self.assertRaises(ValueError):
            init_ascend_config(test_vllm_config)

    @_clean_up_ascend_config
    @patch("vllm_ascend.platform.NPUPlatform._fix_incompatible_config")
    def test_init_ascend_config_enable_npugraph_ex(self, mock_fix_incompatible_config):
        test_vllm_config = VllmConfig()
        test_vllm_config.additional_config = {
            "ascend_compilation_config": {
                "enable_npugraph_ex": True,
                "enable_static_kernel": True
            },
            "refresh": True
        }
        ascend_compilation_config = init_ascend_config(
            test_vllm_config).ascend_compilation_config
        self.assertTrue(ascend_compilation_config.enable_npugraph_ex)
        self.assertTrue(ascend_compilation_config.enable_static_kernel)

    @_clean_up_ascend_config
    @patch("vllm_ascend.platform.NPUPlatform._fix_incompatible_config")
    def test_get_ascend_config(self, mock_fix_incompatible_config):
        test_vllm_config = VllmConfig()
        ascend_config = init_ascend_config(test_vllm_config)
        self.assertEqual(get_ascend_config(), ascend_config)

    @_clean_up_ascend_config
    def test_get_ascend_config_without_init(self):
        with self.assertRaises(RuntimeError):
            get_ascend_config()

    @_clean_up_ascend_config
    @patch("vllm_ascend.platform.NPUPlatform._fix_incompatible_config")
    def test_clear_ascend_config(self, mock_fix_incompatible_config):
        test_vllm_config = VllmConfig()
        ascend_config = init_ascend_config(test_vllm_config)
        self.assertEqual(get_ascend_config(), ascend_config)
        clear_ascend_config()
        with self.assertRaises(RuntimeError):
            get_ascend_config()
