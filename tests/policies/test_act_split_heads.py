#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import (
    ACTPolicy,
    _assemble_lekiwi_banana_action,
    _split_lekiwi_banana_targets,
)
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_STATE


def make_split_head_config() -> ACTConfig:
    config = ACTConfig(
        use_split_heads=True,
        base_mode="binary_move",
        arm_action_dim=6,
        base_direction_index=8,
        base_unused_indices=[6, 7],
        base_forward_speed=30.0,
        chunk_size=4,
        n_action_steps=2,
        use_vae=False,
        dim_model=32,
        dim_feedforward=64,
        n_heads=4,
        n_encoder_layers=1,
        n_decoder_layers=1,
        pretrained_backbone_weights=None,
    )
    config.input_features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(9,)),
        OBS_ENV_STATE: PolicyFeature(type=FeatureType.ENV, shape=(3,)),
    }
    config.output_features = {
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(9,)),
    }
    config.device = "cpu"
    return config


def test_split_lekiwi_banana_targets():
    action = torch.tensor(
        [
            [
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0, 0.0, 0.0],
                [7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 0.0, 0.0, 30.0],
            ]
        ]
    )

    arm_target, base_move_target = _split_lekiwi_banana_targets(action, arm_action_dim=6, base_direction_index=8)

    assert arm_target.shape == (1, 2, 6)
    assert torch.equal(base_move_target, torch.tensor([[[0.0], [1.0]]]))


def test_assemble_lekiwi_banana_action():
    arm_action = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]])
    base_move_prob = torch.tensor([[[0.75]]])

    action = _assemble_lekiwi_banana_action(
        arm_action,
        base_move_prob,
        threshold=0.5,
        forward_speed=30.0,
    )

    assert action.shape == (1, 1, 9)
    assert torch.equal(action[..., 6], torch.zeros(1, 1))
    assert torch.equal(action[..., 7], torch.zeros(1, 1))
    assert torch.equal(action[..., 8], torch.full((1, 1), 30.0))


def test_act_policy_binary_move_forward_logs_split_losses():
    config = make_split_head_config()
    policy = ACTPolicy(config)
    batch = {
        OBS_STATE: torch.randn(2, 9),
        OBS_ENV_STATE: torch.randn(2, 3),
        ACTION: torch.randn(2, config.chunk_size, 9),
        "action_arm_target": torch.randn(2, config.chunk_size, 6),
        "base_move_target": torch.randint(0, 2, (2, config.chunk_size, 1), dtype=torch.float32),
        "action_is_pad": torch.zeros(2, config.chunk_size, dtype=torch.bool),
    }

    loss, loss_dict = policy.forward(batch)

    assert torch.isfinite(loss)
    assert "arm_l1_loss" in loss_dict
    assert "base_bce_loss" in loss_dict
