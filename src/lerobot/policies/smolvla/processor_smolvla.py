#!/usr/bin/env python

# Copyright 2025 HuggingFace Inc. team. All rights reserved.
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

from dataclasses import asdict, dataclass, field
from typing import Any

import torch

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAActionHeadConfig, SmolVLAConfig
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    ComplementaryDataProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep,
    TokenizerProcessorStep,
    TransitionKey,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.utils.constants import (
    OBS_STATE,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)

SMOLVLA_ARM_STATE_KEY = "smolvla.arm_passthrough_state"


def _coerce_action_heads(
    action_heads: list[SmolVLAActionHeadConfig | dict[str, Any]],
) -> list[SmolVLAActionHeadConfig]:
    return [
        head if isinstance(head, SmolVLAActionHeadConfig) else SmolVLAActionHeadConfig(**head)
        for head in action_heads
    ]


def _serialize_action_heads(action_heads: list[SmolVLAActionHeadConfig]) -> list[dict[str, Any]]:
    return [asdict(head) for head in action_heads]


def make_smolvla_pre_post_processors(
    config: SmolVLAConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    Constructs pre-processor and post-processor pipelines for the SmolVLA policy.

    The pre-processing pipeline prepares input data for the model by:
    1.  Renaming features to match pretrained configurations.
    2.  Normalizing input and output features based on dataset statistics.
    3.  Adding a batch dimension.
    4.  Ensuring the language task description ends with a newline character.
    5.  Tokenizing the language task description.
    6.  Moving all data to the specified device.

    The post-processing pipeline handles the model's output by:
    1.  Moving data to the CPU.
    2.  Unnormalizing the output actions to their original scale.

    Args:
        config: The configuration object for the SmolVLA policy.
        dataset_stats: A dictionary of statistics for normalization.

    Returns:
        A tuple containing the configured pre-processor and post-processor pipelines.
    """

    extract_step = None
    input_steps = [
        RenameObservationsProcessorStep(rename_map={}),  # To mimic the same processor as pretrained one
        AddBatchDimensionProcessorStep(),
        SmolVLANewLineProcessor(),
        TokenizerProcessorStep(
            tokenizer_name=config.vlm_model_name,
            padding=config.pad_language_to,
            padding_side="right",
            max_length=config.tokenizer_max_length,
        ),
    ]

    if config.use_discrete_base_heads:
        extract_step = ExtractDiscreteBaseTargetsProcessorStep(action_heads=config.action_heads)
        input_steps.append(extract_step)

    input_steps.extend(
        [
            DeviceProcessorStep(device=config.device),
            NormalizerProcessorStep(
                features={**config.input_features, **config.output_features},
                norm_map=config.normalization_mapping,
                stats=dataset_stats,
            ),
        ]
    )

    output_steps = []

    if config.use_discrete_base_heads:
        output_steps.append(
            AssembleLeKiwiPassthroughActionProcessorStep(
                arm_passthrough_dims=config.arm_passthrough_dims,
                base_action_dims=config.base_action_dims,
                export_action_dim=config.export_action_dim,
                action_heads=config.action_heads,
                extract_step=extract_step,
            )
        )
    else:
        output_steps.append(
            UnnormalizerProcessorStep(
                features=config.output_features, norm_map=config.normalization_mapping, stats=dataset_stats
            )
        )

    # Move to CPU as last step
    output_steps.append(DeviceProcessorStep(device="cpu"))
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


@ProcessorStepRegistry.register(name="smolvla_new_line_processor")
class SmolVLANewLineProcessor(ComplementaryDataProcessorStep):
    """
    A processor step that ensures the 'task' description ends with a newline character.

    This step is necessary for certain tokenizers (e.g., PaliGemma) that expect a
    newline at the end of the prompt. It handles both single string tasks and lists
    of string tasks.
    """

    def complementary_data(self, complementary_data):
        if "task" not in complementary_data:
            return complementary_data

        task = complementary_data["task"]
        if task is None:
            return complementary_data

        new_complementary_data = dict(complementary_data)

        # Handle both string and list of strings
        if isinstance(task, str):
            # Single string: add newline if not present
            if not task.endswith("\n"):
                new_complementary_data["task"] = f"{task}\n"
        elif isinstance(task, list) and all(isinstance(t, str) for t in task):
            # List of strings: add newline to each if not present
            new_complementary_data["task"] = [t if t.endswith("\n") else f"{t}\n" for t in task]
        # If task is neither string nor list of strings, leave unchanged

        return new_complementary_data

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@dataclass
@ProcessorStepRegistry.register(name="extract_discrete_base_targets")
class ExtractDiscreteBaseTargetsProcessorStep(ProcessorStep):
    action_heads: list[SmolVLAActionHeadConfig]
    _last_state: torch.Tensor | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.action_heads = _coerce_action_heads(self.action_heads)

    def __call__(self, transition):
        new_transition = dict(transition)
        observation = new_transition.get(TransitionKey.OBSERVATION) or {}
        complementary_data = dict(new_transition.get(TransitionKey.COMPLEMENTARY_DATA, {}))

        state = observation.get(OBS_STATE) if isinstance(observation, dict) else None
        if state is not None:
            self._last_state = state
            complementary_data[SMOLVLA_ARM_STATE_KEY] = state

        action = new_transition.get(TransitionKey.ACTION)
        if action is None:
            if complementary_data:
                new_transition[TransitionKey.COMPLEMENTARY_DATA] = complementary_data
            return new_transition

        action = torch.as_tensor(action)
        for head in self.action_heads:
            values = action.new_tensor(head.values)
            raw_value = action[..., head.index]
            class_id = torch.argmin(torch.abs(raw_value.unsqueeze(-1) - values), dim=-1)
            complementary_data[f"{head.name}_target"] = class_id.to(torch.long)
        new_transition[TransitionKey.COMPLEMENTARY_DATA] = complementary_data
        return new_transition

    def transform_features(self, features):
        return features

    def get_config(self) -> dict[str, Any]:
        return {"action_heads": _serialize_action_heads(self.action_heads)}


@dataclass
@ProcessorStepRegistry.register(name="assemble_lekiwi_passthrough_action")
class AssembleLeKiwiPassthroughActionProcessorStep(ProcessorStep):
    arm_passthrough_dims: list[int]
    base_action_dims: list[int]
    export_action_dim: int
    action_heads: list[SmolVLAActionHeadConfig]
    extract_step: ExtractDiscreteBaseTargetsProcessorStep | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.action_heads = _coerce_action_heads(self.action_heads)

    def __call__(self, transition):
        head_outputs = transition[TransitionKey.ACTION]
        if isinstance(head_outputs, torch.Tensor):
            return transition
        if not isinstance(head_outputs, dict):
            raise ValueError(f"Expected action to be tensor or dict, got {type(head_outputs)}")

        complementary_data = transition.get(TransitionKey.COMPLEMENTARY_DATA) or {}
        state = complementary_data.get(SMOLVLA_ARM_STATE_KEY)
        if state is None and self.extract_step is not None:
            state = self.extract_step._last_state
        if state is None:
            raise RuntimeError(
                "AssembleLeKiwiPassthroughActionProcessorStep requires cached raw state from "
                "ExtractDiscreteBaseTargetsProcessorStep before it can restore arm passthrough dims."
            )

        first_logits = head_outputs[f"{self.action_heads[0].name}_logits"]
        prefix_shape = torch.argmax(first_logits, dim=-1).shape
        action = state.new_zeros(*prefix_shape, self.export_action_dim)
        passthrough = state[..., self.arm_passthrough_dims]
        while passthrough.ndim < action.ndim:
            passthrough = passthrough.unsqueeze(-2)
        action[..., self.arm_passthrough_dims] = passthrough.expand(
            *prefix_shape, len(self.arm_passthrough_dims)
        )
        for head in self.action_heads:
            logits = head_outputs[f"{head.name}_logits"]
            class_id = torch.argmax(logits, dim=-1)
            values = action.new_tensor(head.values)
            action[..., head.index] = values[class_id]
        new_transition = dict(transition)
        new_transition[TransitionKey.ACTION] = action
        return new_transition

    def transform_features(self, features):
        return features

    def get_config(self) -> dict[str, Any]:
        return {
            "arm_passthrough_dims": self.arm_passthrough_dims,
            "base_action_dims": self.base_action_dims,
            "export_action_dim": self.export_action_dim,
            "action_heads": _serialize_action_heads(self.action_heads),
        }
