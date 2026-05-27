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

import json
from pathlib import Path

from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.factory import make_policy_config


def test_train_pipeline_config_defaults_tensorboard_dir(tmp_path):
    output_dir = tmp_path / "run"
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="user/repo", episodes=[0]),
        policy=make_policy_config("act", push_to_hub=False),
        job_name="banana-act",
        output_dir=output_dir,
    )

    cfg.validate()

    assert cfg.tensorboard_dir == Path("logs") / "banana-act"


def test_train_pipeline_config_defaults_train_log_file(tmp_path):
    output_dir = tmp_path / "run"
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="user/repo", episodes=[0]),
        policy=make_policy_config("act", push_to_hub=False),
        job_name="banana-act",
        output_dir=output_dir,
    )

    cfg.validate()

    assert cfg.train_log_file == Path("logs") / "banana-act.log"


def test_train_pipeline_config_preserves_tensorboard_dir(tmp_path):
    output_dir = tmp_path / "run"
    tensorboard_dir = tmp_path / "custom-tensorboard"
    train_log_file = tmp_path / "custom.log"
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="user/repo", episodes=[0]),
        policy=make_policy_config("act", push_to_hub=False),
        output_dir=output_dir,
        tensorboard_dir=tensorboard_dir,
        train_log_file=train_log_file,
    )

    cfg.validate()

    assert cfg.tensorboard_dir == tensorboard_dir
    assert cfg.train_log_file == train_log_file


def test_train_pipeline_config_from_pretrained_accepts_jsonc_comments(tmp_path):
    output_dir = tmp_path / "run"
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="user/repo", episodes=[0]),
        policy=make_policy_config("act", push_to_hub=False),
        job_name="banana-act",
        output_dir=output_dir,
    )
    config_path = tmp_path / "train_config.annotated.jsonc"
    config_text = json.dumps(cfg.to_dict(), indent=4)
    config_text = config_text.replace(
        '"dataset": {',
        '// dataset comments should be ignored\n    "dataset": {',
        1,
    )
    config_text = config_text.replace(
        '"policy": {',
        '// policy comments should be ignored\n    "policy": {',
        1,
    )
    config_path.write_text(config_text)

    loaded_cfg = TrainPipelineConfig.from_pretrained(config_path)

    assert loaded_cfg.dataset.repo_id == "user/repo"
    assert loaded_cfg.job_name == "banana-act"


def test_train_pipeline_config_from_pretrained_ignores_comment_fields(tmp_path):
    output_dir = tmp_path / "run"
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="user/repo", episodes=[0]),
        policy=make_policy_config("act", push_to_hub=False),
        job_name="banana-act",
        output_dir=output_dir,
    )
    config_path = tmp_path / "train_config.annotated.json"
    config = cfg.to_dict()
    config["_comment"] = "top level comment"
    config["dataset"]["_comment"] = "dataset comment"
    config["policy"]["_comment"] = "policy comment"
    config_path.write_text(json.dumps(config, indent=4))

    loaded_cfg = TrainPipelineConfig.from_pretrained(config_path)

    assert loaded_cfg.dataset.repo_id == "user/repo"
    assert loaded_cfg.job_name == "banana-act"
