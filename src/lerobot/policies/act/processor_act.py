#!/usr/bin/env python

# Copyright 2024 Tony Z. Zhao and The HuggingFace Inc. team. All rights reserved.
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
from dataclasses import dataclass
from typing import Any

import torch

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTActionHeadConfig, ACTConfig
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    RenameObservationsProcessorStep,
    TransitionKey,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME


@dataclass
class ExtractActionHeadTargetsProcessorStep(ProcessorStep):
    action_heads: list[ACTActionHeadConfig]

    def __post_init__(self):
        self.action_heads = [
            head if isinstance(head, ACTActionHeadConfig) else ACTActionHeadConfig(**head)
            for head in self.action_heads
        ]

    def __call__(self, transition):
        new_transition = transition.copy()
        action = new_transition.get(TransitionKey.ACTION)
        if action is None:
            return new_transition

        action_tensor = torch.as_tensor(action)
        complementary_data = dict(new_transition.get(TransitionKey.COMPLEMENTARY_DATA, {}))
        for head in self.action_heads:
            if head.type == "continuous":
                continue
            if head.values is None:
                continue

            raw_value = action_tensor[..., head.indices[0]]
            discrete_values = raw_value.new_tensor(head.values)
            class_ids = torch.argmin(torch.abs(raw_value.unsqueeze(-1) - discrete_values), dim=-1)
            target_key = f"{head.name}_target"
            if head.type == "binary":
                target = class_ids.to(torch.float32).unsqueeze(-1)
            else:
                target = class_ids.to(torch.long)
            complementary_data[target_key] = target
            new_transition[target_key] = target

        new_transition[TransitionKey.COMPLEMENTARY_DATA] = complementary_data
        return new_transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features

    def get_config(self) -> dict[str, Any]:
        return {
            "action_heads": [
                {
                    "name": head.name,
                    "type": head.type,
                    "indices": head.indices,
                    "values": head.values,
                    "loss_weight": head.loss_weight,
                    "threshold": head.threshold,
                    "pos_weight": head.pos_weight,
                }
                for head in self.action_heads
            ]
        }


@dataclass
class ExtractLeKiwiBananaTargetsProcessorStep(ExtractActionHeadTargetsProcessorStep):
    arm_action_dim: int = 6
    base_direction_index: int = 8

    def __init__(self, arm_action_dim: int = 6, base_direction_index: int = 8):
        super().__init__(
            action_heads=[
                ACTActionHeadConfig(
                    name="base_move",
                    type="binary",
                    indices=[base_direction_index],
                    values=[0.0, 30.0],
                )
            ]
        )
        self.arm_action_dim = arm_action_dim
        self.base_direction_index = base_direction_index

    def get_config(self) -> dict[str, Any]:
        return {
            "arm_action_dim": self.arm_action_dim,
            "base_direction_index": self.base_direction_index,
        }


def make_act_pre_post_processors(
    config: ACTConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Creates the pre- and post-processing pipelines for the ACT policy.

    The pre-processing pipeline handles normalization, batching, and device placement for the model inputs.
    The post-processing pipeline handles unnormalization and moves the model outputs back to the CPU.

    Args:
        config (ACTConfig): The ACT policy configuration object.
        dataset_stats (dict[str, dict[str, torch.Tensor]] | None): A dictionary containing dataset
            statistics (e.g., mean and std) used for normalization. Defaults to None.

    Returns:
        tuple[PolicyProcessorPipeline[dict[str, Any], dict[str, Any]], PolicyProcessorPipeline[PolicyAction, PolicyAction]]: A tuple containing the
        pre-processor pipeline and the post-processor pipeline.
    """

    input_steps = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
    ]
    if config.use_split_heads:
        if config.base_mode == "binary_move":
            input_steps.append(
                ExtractLeKiwiBananaTargetsProcessorStep(
                    arm_action_dim=config.arm_action_dim,
                    base_direction_index=config.base_direction_index,
                )
            )
        elif config.action_heads:
            input_steps.append(
                ExtractActionHeadTargetsProcessorStep(
                    action_heads=config.action_heads,
                )
            )
    input_steps.extend(
        [
            DeviceProcessorStep(device=config.device),
            NormalizerProcessorStep(
                features={**config.input_features, **config.output_features},
                norm_map=config.normalization_mapping,
                stats=dataset_stats,
                device=config.device,
            ),
        ]
    )
    output_steps = [
        UnnormalizerProcessorStep(
            features=config.output_features, norm_map=config.normalization_mapping, stats=dataset_stats
        ),
        DeviceProcessorStep(device="cpu"),
    ]

    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
