# ruff: noqa: E402, I001
import importlib
import json
import sys
from pathlib import Path

import torch

_repo_root = Path(__file__).resolve().parents[2]
_src = _repo_root / "src"
_src_str = str(_src)
if _src_str not in sys.path:
    sys.path.insert(0, _src_str)

for m in list(sys.modules):
    if m == "lerobot" or m.startswith("lerobot."):
        del sys.modules[m]
importlib.invalidate_caches()

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAActionHeadConfig, SmolVLAConfig
from lerobot.policies.smolvla.processor_smolvla import (
    AssembleLeKiwiPassthroughActionProcessorStep,
    ExtractDiscreteBaseTargetsProcessorStep,
    SMOLVLA_ARM_STATE_KEY,
    SmolVLANewLineProcessor,
)
from lerobot.processor import NormalizerProcessorStep, ProcessorStepRegistry, TransitionKey
from lerobot.utils.constants import ACTION, OBS_STATE


def test_smolvla_mixed_action_processors_remain_registered():
    assert ProcessorStepRegistry.get("smolvla_new_line_processor") is SmolVLANewLineProcessor
    assert (
        ProcessorStepRegistry.get("extract_discrete_base_targets") is ExtractDiscreteBaseTargetsProcessorStep
    )
    assert (
        ProcessorStepRegistry.get("assemble_lekiwi_passthrough_action")
        is AssembleLeKiwiPassthroughActionProcessorStep
    )


def test_extract_discrete_base_targets_maps_three_classes_and_caches_raw_state():
    step = ExtractDiscreteBaseTargetsProcessorStep(
        action_heads=[
            SmolVLAActionHeadConfig("base_x", "categorical", 6, [-0.25, 0.0, 0.25]),
            SmolVLAActionHeadConfig("base_y", "categorical", 7, [-0.25, 0.0, 0.25]),
            SmolVLAActionHeadConfig("base_theta", "categorical", 8, [-0.25, 0.0, 0.25]),
        ]
    )
    state = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0, 0.0, 0.0])
    transition = {
        TransitionKey.OBSERVATION: {"observation.state": state},
        TransitionKey.ACTION: torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, -0.25, 0.0]),
        TransitionKey.COMPLEMENTARY_DATA: {},
    }

    out = step(transition)

    assert torch.equal(out[TransitionKey.COMPLEMENTARY_DATA]["base_x_target"], torch.tensor(2))
    assert torch.equal(out[TransitionKey.COMPLEMENTARY_DATA]["base_y_target"], torch.tensor(0))
    assert torch.equal(out[TransitionKey.COMPLEMENTARY_DATA]["base_theta_target"], torch.tensor(1))
    assert torch.equal(out[TransitionKey.COMPLEMENTARY_DATA][SMOLVLA_ARM_STATE_KEY], state)
    assert torch.equal(step._last_state, state)


def test_assemble_passthrough_action_restores_nine_dims_from_cached_state():
    extract_step = ExtractDiscreteBaseTargetsProcessorStep(
        action_heads=[
            SmolVLAActionHeadConfig("base_x", "categorical", 6, [-0.25, 0.0, 0.25]),
            SmolVLAActionHeadConfig("base_y", "categorical", 7, [-0.25, 0.0, 0.25]),
            SmolVLAActionHeadConfig("base_theta", "categorical", 8, [-0.25, 0.0, 0.25]),
        ]
    )
    state = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0, 0.0, 0.0])
    extract_step(
        {
            TransitionKey.OBSERVATION: {"observation.state": state},
            TransitionKey.COMPLEMENTARY_DATA: {},
        }
    )

    step = AssembleLeKiwiPassthroughActionProcessorStep(
        arm_passthrough_dims=[0, 1, 2, 3, 4, 5],
        base_action_dims=[6, 7, 8],
        export_action_dim=9,
        action_heads=[
            SmolVLAActionHeadConfig("base_x", "categorical", 6, [-0.25, 0.0, 0.25]),
            SmolVLAActionHeadConfig("base_y", "categorical", 7, [-0.25, 0.0, 0.25]),
            SmolVLAActionHeadConfig("base_theta", "categorical", 8, [-0.25, 0.0, 0.25]),
        ],
        extract_step=extract_step,
    )
    transition = {
        TransitionKey.ACTION: {
            "base_x_logits": torch.tensor([[0.0, 0.0, 4.0], [0.0, 0.0, 4.0]]),
            "base_y_logits": torch.tensor([[4.0, 0.0, 0.0], [4.0, 0.0, 0.0]]),
            "base_theta_logits": torch.tensor([[0.0, 5.0, 0.0], [0.0, 5.0, 0.0]]),
        },
    }

    out = step(transition)

    assert torch.equal(
        out[TransitionKey.ACTION],
        torch.tensor(
            [
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.25, -0.25, 0.0],
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.25, -0.25, 0.0],
            ]
        ),
    )


def test_make_pre_post_processors_loads_legacy_empty_smolvla_discrete_base_configs(tmp_path):
    config = SmolVLAConfig(
        use_discrete_base_heads=True,
        arm_passthrough_dims=[0, 1, 2, 3, 4, 5],
        export_action_dim=9,
    )
    config.input_features = {OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(9,))}
    config.output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(9,))}
    config.normalization_mapping = {
        FeatureType.STATE: NormalizationMode.MEAN_STD,
        FeatureType.ACTION: NormalizationMode.MEAN_STD,
    }
    stats = {
        OBS_STATE: {"mean": torch.zeros(9), "std": torch.ones(9)},
        ACTION: {"mean": torch.zeros(9), "std": torch.ones(9)},
    }

    features = {
        OBS_STATE: {"type": "STATE", "shape": [9]},
        ACTION: {"type": "ACTION", "shape": [9]},
    }
    norm_map = {"STATE": "MEAN_STD", "ACTION": "MEAN_STD"}
    (tmp_path / "policy_preprocessor.json").write_text(
        json.dumps(
            {
                "name": "policy_preprocessor",
                "steps": [
                    {"registry_name": "extract_discrete_base_targets", "config": {}},
                    {
                        "registry_name": "normalizer_processor",
                        "config": {"features": features, "norm_map": norm_map},
                    },
                ],
            }
        )
    )
    (tmp_path / "policy_postprocessor.json").write_text(
        json.dumps(
            {
                "name": "policy_postprocessor",
                "steps": [
                    {"registry_name": "assemble_lekiwi_passthrough_action", "config": {}},
                ],
            }
        )
    )

    preprocessor, postprocessor = make_pre_post_processors(
        config,
        pretrained_path=tmp_path,
        dataset_stats=stats,
    )

    extract_step = preprocessor.steps[0]
    normalizer_step = preprocessor.steps[1]
    assemble_step = postprocessor.steps[0]

    assert isinstance(extract_step, ExtractDiscreteBaseTargetsProcessorStep)
    assert isinstance(normalizer_step, NormalizerProcessorStep)
    assert isinstance(assemble_step, AssembleLeKiwiPassthroughActionProcessorStep)
    assert assemble_step.extract_step is extract_step
    assert [head.name for head in extract_step.action_heads] == ["base_x", "base_y", "base_theta"]
    assert assemble_step.arm_passthrough_dims == [0, 1, 2, 3, 4, 5]
    assert assemble_step.base_action_dims == [6, 7, 8]
