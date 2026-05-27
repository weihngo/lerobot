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

import importlib.util
import re
from pathlib import Path

import torch


def load_evaluate_module():
    module_path = Path("/home/lwh/code/lerobot/examples/lekiwi/evaluate.py")
    spec = importlib.util.spec_from_file_location("examples.lekiwi.evaluate", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_resolve_policy_stats_uses_eval_dataset_stats_by_default():
    module = load_evaluate_module()

    eval_stats = {"action": {"mean": "eval"}}

    stats = module.resolve_policy_stats(
        eval_dataset_stats=eval_stats,
        train_dataset_repo_id=None,
        train_dataset_root=None,
    )

    assert stats is eval_stats


def test_resolve_policy_stats_uses_training_dataset_stats_when_configured(monkeypatch):
    module = load_evaluate_module()

    class FakeMetadata:
        def __init__(self, repo_id, root=None):
            assert repo_id == "lekiwi_pick_and_put_banana"
            assert root == Path("/mnt/data/yzh/dataset/lekiwi_pick_and_put_banana")
            self.stats = {"action": {"mean": "train"}}

    monkeypatch.setattr(module, "LeRobotDatasetMetadata", FakeMetadata)

    stats = module.resolve_policy_stats(
        eval_dataset_stats={"action": {"mean": "eval"}},
        train_dataset_repo_id="lekiwi_pick_and_put_banana",
        train_dataset_root=Path("/mnt/data/yzh/dataset/lekiwi_pick_and_put_banana"),
    )

    assert stats == {"action": {"mean": "train"}}


def test_resolve_policy_stats_applies_training_imagenet_camera_stats(monkeypatch):
    module = load_evaluate_module()

    class FakeMetadata:
        def __init__(self, repo_id, root=None):
            assert repo_id == "lekiwi_pick_and_put_banana"
            assert root == Path("/mnt/data/yzh/dataset/lekiwi_pick_and_put_banana")
            self.camera_keys = ["observation.images.front"]
            self.stats = {
                "observation.images.front": {"mean": "train-front", "std": "train-front-std"},
                "action": {"mean": "train"},
            }

    monkeypatch.setattr(module, "LeRobotDatasetMetadata", FakeMetadata)
    monkeypatch.setattr(
        module,
        "IMAGENET_STATS",
        {"mean": [0.1, 0.2, 0.3], "std": [0.4, 0.5, 0.6]},
        raising=False,
    )

    stats = module.resolve_policy_stats(
        eval_dataset_stats={"action": {"mean": "eval"}},
        train_dataset_repo_id="lekiwi_pick_and_put_banana",
        train_dataset_root=Path("/mnt/data/yzh/dataset/lekiwi_pick_and_put_banana"),
    )

    assert torch.equal(stats["observation.images.front"]["mean"], torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32))
    assert torch.equal(stats["observation.images.front"]["std"], torch.tensor([0.4, 0.5, 0.6], dtype=torch.float32))
    assert stats["action"] == {"mean": "train"}


def test_resolve_control_log_path_uses_job_name_from_checkpoint_path():
    module = load_evaluate_module()

    log_path = module.resolve_control_log_path(
        "/home/lwh/code/lerobot/outputs/train/act_lekiwi_banana_binary_base/checkpoints/last/pretrained_model"
    )

    assert log_path.parent == Path("logs/act_lekiwi_banana_binary_base")
    assert re.fullmatch(r"lekiwi_control_\d{8}_\d{6}\.txt", log_path.name)


def test_append_control_log_writes_one_line_per_action(tmp_path):
    module = load_evaluate_module()
    log_path = tmp_path / "lekiwi_control.txt"

    module.append_control_log(
        log_path,
        {
            "arm_shoulder_pan.pos": 1.0,
            "theta.vel": 30.0,
        },
    )

    lines = log_path.read_text().strip().splitlines()

    assert len(lines) == 1
    assert "arm_shoulder_pan.pos=1.0" in lines[0]
    assert "theta.vel=30.0" in lines[0]


def test_load_policy_for_evaluate_passes_dataset_stats_to_act_policy(monkeypatch):
    module = load_evaluate_module()
    captured = {}

    class FakePolicy:
        pass

    def fake_from_pretrained(path, dataset_stats=None):
        captured["path"] = path
        captured["dataset_stats"] = dataset_stats
        return FakePolicy()

    monkeypatch.setattr(module.ACTPolicy, "from_pretrained", fake_from_pretrained)

    stats = {"action": {"mean": "train"}}
    policy = module.load_policy_for_evaluate(
        pretrained_path="/tmp/pretrained_model",
        dataset_stats=stats,
    )

    assert isinstance(policy, FakePolicy)
    assert captured["path"] == "/tmp/pretrained_model"
    assert captured["dataset_stats"] is stats
