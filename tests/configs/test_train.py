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

from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.factory import make_policy_config


def test_train_pipeline_config_defaults_tensorboard_dir(tmp_path):
    output_dir = tmp_path / "run"
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="user/repo", episodes=[0]),
        policy=make_policy_config("act", push_to_hub=False),
        output_dir=output_dir,
    )

    cfg.validate()

    assert cfg.tensorboard_dir == output_dir / "tensorboard"


def test_train_pipeline_config_preserves_tensorboard_dir(tmp_path):
    output_dir = tmp_path / "run"
    tensorboard_dir = tmp_path / "custom-tensorboard"
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="user/repo", episodes=[0]),
        policy=make_policy_config("act", push_to_hub=False),
        output_dir=output_dir,
        tensorboard_dir=tensorboard_dir,
    )

    cfg.validate()

    assert cfg.tensorboard_dir == tensorboard_dir
