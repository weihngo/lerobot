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

import csv
import importlib.util
import json
import re
import sys
import types
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

try:
    import torch
except ModuleNotFoundError:
    class _FakeTensor:
        def __init__(self, value):
            self.value = value

        def item(self):
            if isinstance(self.value, list):
                current = self.value
                while isinstance(current, list):
                    current = current[0]
                return current
            return self.value

    class _FakeDevice:
        def __init__(self, name):
            self.type = str(name)

    def _wrap(value):
        return _FakeTensor(value)

    torch = types.SimpleNamespace(
        Tensor=_FakeTensor,
        tensor=lambda value, dtype=None: _wrap(value),
        zeros=lambda *shape, **kwargs: _wrap(0),
        ones=lambda *shape, **kwargs: _wrap(1),
        full=lambda shape, fill_value, **kwargs: _wrap(fill_value),
        device=lambda name: _FakeDevice(name),
        inference_mode=lambda: nullcontext(),
        autocast=lambda **kwargs: nullcontext(),
        float32="float32",
        uint8="uint8",
    )


def load_inspect_module():
    module_path = REPO_ROOT / "examples" / "lekiwi" / "inspect_frame_inference.py"
    spec = importlib.util.spec_from_file_location("examples.lekiwi.inspect_frame_inference", module_path)
    module = importlib.util.module_from_spec(spec)
    lerobot_module = types.ModuleType("lerobot")
    configs_module = types.ModuleType("lerobot.configs")
    policies_config_module = types.ModuleType("lerobot.configs.policies")
    datasets_module = types.ModuleType("lerobot.datasets")
    datasets_factory_module = types.ModuleType("lerobot.datasets.factory")
    dataset_metadata_module = types.ModuleType("lerobot.datasets.dataset_metadata")
    lerobot_dataset_module = types.ModuleType("lerobot.datasets.lerobot_dataset")
    policies_module = types.ModuleType("lerobot.policies")
    policies_factory_module = types.ModuleType("lerobot.policies.factory")
    utils_module = types.ModuleType("lerobot.utils")
    constants_module = types.ModuleType("lerobot.utils.constants")
    pil_module = types.ModuleType("PIL")
    pil_image_module = types.ModuleType("PIL.Image")

    class StubPreTrainedConfig:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            return cls()

    class StubDatasetMetadata:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class StubDataset:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    def _unexpected_factory_call(*_args, **_kwargs):
        raise AssertionError("Test should patch policy factory helpers before use.")

    class _StubImageFile:
        def save(self, path):
            Path(path).write_bytes(b"stub")

    class _StubImage:
        @staticmethod
        def fromarray(_array):
            return _StubImageFile()

    policies_config_module.PreTrainedConfig = StubPreTrainedConfig
    datasets_factory_module.IMAGENET_STATS = {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    }
    dataset_metadata_module.LeRobotDatasetMetadata = StubDatasetMetadata
    lerobot_dataset_module.LeRobotDataset = StubDataset
    policies_factory_module.make_policy = _unexpected_factory_call
    policies_factory_module.make_pre_post_processors = _unexpected_factory_call
    constants_module.ACTION = "action"
    pil_image_module.Image = _StubImage
    pil_image_module.fromarray = _StubImage.fromarray
    pil_module.Image = pil_image_module

    stubs = {
        "PIL": pil_module,
        "PIL.Image": pil_image_module,
        "lerobot": lerobot_module,
        "lerobot.configs": configs_module,
        "lerobot.configs.policies": policies_config_module,
        "lerobot.datasets": datasets_module,
        "lerobot.datasets.factory": datasets_factory_module,
        "lerobot.datasets.dataset_metadata": dataset_metadata_module,
        "lerobot.datasets.lerobot_dataset": lerobot_dataset_module,
        "lerobot.policies": policies_module,
        "lerobot.policies.factory": policies_factory_module,
        "lerobot.utils": utils_module,
        "lerobot.utils.constants": constants_module,
    }
    if isinstance(torch, types.SimpleNamespace):
        stubs["torch"] = torch
    assert spec.loader is not None
    with patch.dict(sys.modules, stubs):
        spec.loader.exec_module(module)
    return module


class FakeDataset:
    def __init__(self):
        self.num_frames = 25
        self.meta = type(
            "Meta",
            (),
            {
                "episodes": [
                    {"dataset_from_index": 0, "dataset_to_index": 10},
                    {"dataset_from_index": 10, "dataset_to_index": 25},
                ]
            },
        )()


def _make_rgb_frame(rgb: tuple[float, float, float]) -> torch.Tensor:
    return torch.tensor(rgb, dtype=torch.float32).view(3, 1, 1).repeat(1, 4, 5)


def _read_video_frames(video_path: Path):
    torchvision = pytest.importorskip("torchvision")
    if not hasattr(torchvision, "io") or not hasattr(torchvision.io, "read_video"):
        pytest.skip("torchvision.io.read_video is unavailable in this environment")
    frames, _, _ = torchvision.io.read_video(str(video_path), pts_unit="sec")
    return frames


def _assert_decodable_clip_matches_expected_order(
    video_path: Path, expected_frame_values: list[tuple[float, float, float]]
):
    frames = _read_video_frames(video_path)

    assert tuple(frames.shape) == (len(expected_frame_values), 4, 5, 3)

    decoded_means = frames.to(torch.float32).mean(dim=(1, 2))
    expected_means = torch.tensor(expected_frame_values, dtype=torch.float32) * 255
    assert torch.allclose(decoded_means, expected_means, atol=20.0, rtol=0.0)


def test_resolve_sample_index_from_episode_and_frame():
    module = load_inspect_module()
    dataset = FakeDataset()

    sample_index = module.resolve_sample_index(dataset=dataset, episode_index=1, frame_index=3)

    assert sample_index == 13


def test_resolve_sample_index_returns_global_index_passthrough():
    module = load_inspect_module()
    dataset = FakeDataset()

    sample_index = module.resolve_sample_index(dataset=dataset, global_index=7)

    assert sample_index == 7


def test_resolve_sample_index_rejects_conflicting_index_selectors():
    module = load_inspect_module()
    dataset = FakeDataset()

    with pytest.raises(ValueError, match="Provide either global_index or episode_index with frame_index"):
        module.resolve_sample_index(dataset=dataset, global_index=7, episode_index=1, frame_index=3)


@pytest.mark.parametrize(
    ("kwargs", "expected_message"),
    [
        ({}, "Provide either global_index or episode_index with frame_index"),
        ({"episode_index": 1}, "episode_index and frame_index must be provided together"),
        ({"frame_index": 3}, "episode_index and frame_index must be provided together"),
    ],
)
def test_resolve_sample_index_requires_complete_frame_selection_inputs(kwargs, expected_message):
    module = load_inspect_module()
    dataset = FakeDataset()

    with pytest.raises(ValueError, match=expected_message):
        module.resolve_sample_index(dataset=dataset, **kwargs)


@pytest.mark.parametrize(
    ("kwargs", "expected_message"),
    [
        ({"global_index": -1}, "global_index -1 is out of bounds"),
        ({"global_index": 25}, "global_index 25 is out of bounds"),
        ({"episode_index": -1, "frame_index": 0}, "episode_index -1 is out of bounds"),
        ({"episode_index": 2, "frame_index": 0}, "episode_index 2 is out of bounds"),
        ({"episode_index": 1, "frame_index": -1}, "frame_index -1 is out of bounds"),
        ({"episode_index": 1, "frame_index": 15}, "frame_index 15 is out of bounds"),
    ],
)
def test_resolve_sample_index_rejects_out_of_bounds_indices(kwargs, expected_message):
    module = load_inspect_module()
    dataset = FakeDataset()

    with pytest.raises(IndexError, match=expected_message):
        module.resolve_sample_index(dataset=dataset, **kwargs)


def test_extract_inference_inputs_adds_batch_dimension_and_metadata():
    module = load_inspect_module()
    sample = {
        "observation.images.front": torch.arange(60, dtype=torch.float32).reshape(3, 4, 5),
        "observation.images.wrist": torch.arange(60, dtype=torch.float32).reshape(3, 4, 5),
        "observation.state": torch.tensor([1.0, 2.0, 3.0]),
        "action": torch.tensor([0.5, -0.25]),
        "episode_index": torch.tensor(2),
        "frame_index": torch.tensor(6),
        "index": torch.tensor(26),
        "task": "pick banana",
    }

    observation, label, metadata = module.extract_inference_inputs(
        sample=sample,
        input_keys=["observation.images.front", "observation.images.wrist", "observation.state"],
        device=torch.device("cpu"),
    )

    assert set(observation) == {"observation.images.front", "observation.images.wrist", "observation.state", "task"}
    assert observation["observation.images.front"].shape == (1, 3, 4, 5)
    assert observation["observation.images.wrist"].shape == (1, 3, 4, 5)
    assert observation["observation.state"].shape == (1, 3)
    assert label.shape == (2,)
    assert metadata == {"episode_index": 2, "frame_index": 6, "global_index": 26}


@pytest.mark.parametrize(
    ("missing_key", "expected_message"),
    [
        ("observation.images.front", "Missing required observation key: observation.images.front"),
        ("observation.images.wrist", "Missing required observation key: observation.images.wrist"),
        ("action", "Missing required label key: action"),
    ],
)
def test_extract_inference_inputs_rejects_missing_required_sample_keys(missing_key, expected_message):
    module = load_inspect_module()
    sample = {
        "observation.images.front": torch.arange(60, dtype=torch.float32).reshape(3, 4, 5),
        "observation.images.wrist": torch.arange(60, dtype=torch.float32).reshape(3, 4, 5),
        "observation.state": torch.tensor([1.0, 2.0, 3.0]),
        "action": torch.tensor([0.5, -0.25]),
        "episode_index": torch.tensor(2),
        "frame_index": torch.tensor(6),
        "index": torch.tensor(26),
    }
    sample.pop(missing_key)

    with pytest.raises(KeyError, match=re.escape(expected_message)):
        module.extract_inference_inputs(
            sample=sample,
            input_keys=["observation.images.front", "observation.images.wrist", "observation.state"],
            device=torch.device("cpu"),
        )


def test_save_artifacts_writes_images_and_comparison_files(tmp_path):
    module = load_inspect_module()
    sample = {
        "observation.images.front": torch.zeros(3, 4, 5),
        "observation.images.wrist": torch.ones(3, 4, 5),
    }
    comparison = {
        "policy_path": "pretrained_model",
        "dataset_repo_id": "lekiwi_pick_and_put_banana",
        "dataset_root": "dataset",
        "stats_dataset_repo_id": "training_stats_repo",
        "stats_dataset_root": "training_stats_root",
        "episode_index": 1,
        "frame_index": 4,
        "global_index": 10,
        "action_names": ["theta.vel"],
        "prediction": {"theta.vel": 0.0},
        "label": {"theta.vel": 30.0},
        "delta": {"theta.vel": -30.0},
    }

    module.save_artifacts(output_dir=tmp_path, sample=sample, comparison=comparison)

    assert (tmp_path / "front_frame.png").exists()
    assert (tmp_path / "wrist_frame.png").exists()
    assert (tmp_path / "summary.txt").exists()

    saved_comparison = json.loads((tmp_path / "comparison.json").read_text())
    assert saved_comparison == comparison
    assert "theta.vel" in (tmp_path / "summary.txt").read_text()


def test_run_inspection_selects_sample_by_global_index(tmp_path, monkeypatch):
    module = load_inspect_module()
    captured = {}

    class FakeDatasetForGlobalIndex:
        def __init__(self, *args, **kwargs):
            self.meta = type(
                "Meta",
                (),
                {
                    "episodes": [{"dataset_from_index": 0, "dataset_to_index": 3}],
                    "stats": {"action": {"mean": torch.tensor([0.0, 0.0])}},
                    "features": {"action": {"names": ["joint", "theta.vel"]}},
                },
            )()
            self.num_frames = 3

        def __getitem__(self, idx):
            captured["sample_index"] = idx
            return {
                "observation.images.front": torch.zeros(3, 4, 5),
                "observation.images.wrist": torch.ones(3, 4, 5),
                "observation.state": torch.tensor([1.0, 2.0, 3.0]),
                "action": torch.tensor([0.5, -0.25]),
                "episode_index": torch.tensor(0),
                "frame_index": torch.tensor(idx),
                "index": torch.tensor(idx),
                "task": "pick banana",
            }

    class FakePolicy:
        config = type(
            "Config",
            (),
            {
                "device": "cpu",
                "use_amp": False,
                "input_features": {
                    "observation.images.front": object(),
                    "observation.images.wrist": object(),
                    "observation.state": object(),
                },
            },
        )()

        def select_action(self, observation):
            return torch.tensor([[0.0, 0.25]])

    monkeypatch.setattr(module, "LeRobotDataset", FakeDatasetForGlobalIndex)
    monkeypatch.setattr(
        module,
        "load_policy_and_processors",
        lambda **kwargs: (FakePolicy(), lambda batch: batch, lambda action: action),
    )

    result = module.run_inspection(
        policy_path=Path("pretrained_model"),
        dataset_repo_id="lekiwi_pick_and_put_banana",
        dataset_root=Path("dataset"),
        output_dir=tmp_path,
        global_index=2,
    )

    assert captured["sample_index"] == 2
    assert result["global_index"] == 2
    assert result["frame_index"] == 2


def test_run_inspection_builds_prediction_label_and_delta(tmp_path, monkeypatch):
    module = load_inspect_module()

    class FakeDatasetForRun:
        def __init__(self, *args, **kwargs):
            self.meta = type(
                "Meta",
                (),
                {
                    "episodes": [{"dataset_from_index": 0, "dataset_to_index": 1}],
                    "stats": {"action": {"mean": torch.tensor([0.0, 0.0])}},
                    "features": {"action": {"names": ["joint", "theta.vel"]}},
                },
            )()
            self.num_frames = 1

        def __getitem__(self, idx):
            assert idx == 0
            return {
                "observation.images.front": torch.zeros(3, 4, 5),
                "observation.images.wrist": torch.ones(3, 4, 5),
                "observation.state": torch.tensor([1.0, 2.0, 3.0]),
                "action": torch.tensor([0.5, -0.25]),
                "episode_index": torch.tensor(0),
                "frame_index": torch.tensor(0),
                "index": torch.tensor(0),
                "task": "pick banana",
            }

    class FakePolicy:
        config = type(
            "Config",
            (),
            {
                "device": "cpu",
                "use_amp": False,
                "input_features": {
                    "observation.images.front": object(),
                    "observation.images.wrist": object(),
                    "observation.state": object(),
                },
            },
        )()

        def select_action(self, observation):
            assert observation["observation.state"].shape == (1, 3)
            return torch.tensor([[0.0, 0.25]])

    monkeypatch.setattr(module, "LeRobotDataset", FakeDatasetForRun)
    monkeypatch.setattr(
        module,
        "load_policy_and_processors",
        lambda **kwargs: (FakePolicy(), lambda batch: batch, lambda action: action),
    )

    result = module.run_inspection(
        policy_path=Path("/tmp/pretrained_model"),
        dataset_repo_id="lekiwi_pick_and_put_banana",
        dataset_root=Path("/tmp/dataset"),
        output_dir=tmp_path,
        episode_index=0,
        frame_index=0,
    )

    assert result["policy_path"] == "/tmp/pretrained_model"
    assert result["dataset_repo_id"] == "lekiwi_pick_and_put_banana"
    assert result["dataset_root"] == "/tmp/dataset"
    assert result["stats_dataset_repo_id"] == "lekiwi_pick_and_put_banana"
    assert result["stats_dataset_root"] == "/tmp/dataset"
    assert result["episode_index"] == 0
    assert result["frame_index"] == 0
    assert result["global_index"] == 0
    assert result["action_names"] == ["joint", "theta.vel"]
    assert result["prediction"] == {"joint": 0.0, "theta.vel": 0.25}
    assert result["label"] == {"joint": 0.5, "theta.vel": -0.25}
    assert result["delta"] == {"joint": -0.5, "theta.vel": 0.5}
    comparison_json = tmp_path / "comparison.json"
    assert comparison_json.exists()
    assert json.loads(comparison_json.read_text()) == result


def test_run_inspection_compares_only_smolvla_predicted_action_dims(tmp_path, monkeypatch):
    module = load_inspect_module()

    class FakeDatasetForMixedAction:
        def __init__(self, *args, **kwargs):
            self.meta = type(
                "Meta",
                (),
                {
                    "episodes": [{"dataset_from_index": 0, "dataset_to_index": 1}],
                    "stats": {"action": {"mean": torch.zeros(9)}},
                    "features": {
                        "action": {
                            "names": [
                                "arm_0",
                                "arm_1",
                                "arm_2",
                                "arm_3",
                                "arm_4",
                                "arm_5",
                                "x.vel",
                                "y.vel",
                                "theta.vel",
                            ]
                        }
                    },
                },
            )()
            self.num_frames = 1

        def __getitem__(self, idx):
            assert idx == 0
            return {
                "observation.images.front": torch.zeros(3, 4, 5),
                "observation.images.wrist": torch.ones(3, 4, 5),
                "observation.state": torch.arange(9, dtype=torch.float32),
                "action": torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, -0.2, 0.0, 0.3]),
                "episode_index": torch.tensor(0),
                "frame_index": torch.tensor(0),
                "index": torch.tensor(0),
                "task": "pick banana",
            }

    class FakePolicy:
        config = type(
            "Config",
            (),
            {
                "device": "cpu",
                "use_amp": False,
                "use_discrete_base_heads": True,
                "base_action_dims": [6, 7, 8],
                "input_features": {
                    "observation.images.front": object(),
                    "observation.images.wrist": object(),
                    "observation.state": object(),
                },
            },
        )()

        def select_action(self, observation):
            return torch.tensor([[10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 0.1, -0.1, 0.2]])

    monkeypatch.setattr(module, "LeRobotDataset", FakeDatasetForMixedAction)
    monkeypatch.setattr(
        module,
        "load_policy_and_processors",
        lambda **kwargs: (FakePolicy(), lambda batch: batch, lambda action: action),
    )

    result = module.run_inspection(
        policy_path=Path("/tmp/pretrained_model"),
        dataset_repo_id="lekiwi_pick_and_put_banana",
        dataset_root=Path("/tmp/dataset"),
        output_dir=tmp_path,
        global_index=0,
    )

    assert result["action_names"] == ["x.vel", "y.vel", "theta.vel"]
    assert result["predicted_action_indices"] == [6, 7, 8]
    assert result["raw_prediction_dim"] == 9
    assert result["prediction"] == pytest.approx({"x.vel": 0.1, "y.vel": -0.1, "theta.vel": 0.2})
    assert result["label"] == pytest.approx({"x.vel": -0.2, "y.vel": 0.0, "theta.vel": 0.3})
    assert result["delta"] == pytest.approx({"x.vel": 0.3, "y.vel": -0.1, "theta.vel": -0.1})


def test_load_policy_and_processors_disables_pretrained_processors_for_mixed_action(monkeypatch):
    module = load_inspect_module()

    policy_cfg = type(
        "Config",
        (),
        {
            "device": "cpu",
            "pretrained_path": "/tmp/pretrained_model",
            "use_discrete_base_heads": True,
        },
    )()
    dataset_meta = object()
    stats_meta = type("StatsMeta", (), {"stats": {"action": {"mean": torch.zeros(9)}}})()

    captured = {}

    monkeypatch.setattr(module.PreTrainedConfig, "from_pretrained", lambda path: policy_cfg)
    monkeypatch.setattr(module, "make_policy", lambda cfg, ds_meta: type("Policy", (), {"config": cfg})())

    def fake_make_pre_post_processors(**kwargs):
        captured.update(kwargs)
        return object(), object()

    monkeypatch.setattr(module, "make_pre_post_processors", fake_make_pre_post_processors)

    module.load_policy_and_processors(
        policy_path=Path("/tmp/pretrained_model"),
        dataset_meta=dataset_meta,
        stats_meta=stats_meta,
        device=None,
    )

    assert captured["pretrained_path"] is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"stats_dataset_repo_id": "training_stats_repo"},
        {"stats_dataset_root": Path("training_stats_root")},
    ],
)
def test_run_inspection_requires_complete_stats_dataset_source_args(tmp_path, kwargs):
    module = load_inspect_module()

    with pytest.raises(ValueError, match="stats_dataset_repo_id and stats_dataset_root must be provided together"):
        module.run_inspection(
            policy_path=Path("pretrained_model"),
            dataset_repo_id="lekiwi_pick_and_put_banana",
            dataset_root=Path("dataset"),
            output_dir=tmp_path,
            global_index=0,
            **kwargs,
        )


def test_run_inspection_applies_training_imagenet_camera_stats_by_default(tmp_path, monkeypatch):
    module = load_inspect_module()
    captured = {}

    class FakeDatasetForImagenetStats:
        def __init__(self, *args, **kwargs):
            self.meta = type(
                "Meta",
                (),
                {
                    "episodes": [{"dataset_from_index": 0, "dataset_to_index": 1}],
                    "camera_keys": ["observation.images.front", "observation.images.wrist"],
                    "robot_type": "lekiwi",
                    "stats": {
                        "action": {"mean": torch.tensor([0.0, 0.0])},
                        "observation.images.front": {
                            "mean": torch.tensor([[[0.1]], [[0.2]], [[0.3]]]),
                            "std": torch.tensor([[[0.4]], [[0.5]], [[0.6]]]),
                        },
                        "observation.images.wrist": {
                            "mean": torch.tensor([[[0.7]], [[0.8]], [[0.9]]]),
                            "std": torch.tensor([[[1.0]], [[1.1]], [[1.2]]]),
                        },
                    },
                    "features": {"action": {"names": ["joint", "theta.vel"]}},
                },
            )()
            self.num_frames = 1

        def __getitem__(self, idx):
            return {
                "observation.images.front": torch.zeros(3, 4, 5),
                "observation.images.wrist": torch.ones(3, 4, 5),
                "observation.state": torch.tensor([1.0, 2.0, 3.0]),
                "action": torch.tensor([0.5, -0.25]),
                "episode_index": torch.tensor(0),
                "frame_index": torch.tensor(0),
                "index": torch.tensor(0),
                "task": "pick banana",
            }

    class FakePolicy:
        config = type(
            "Config",
            (),
            {
                "device": "cpu",
                "use_amp": False,
                "input_features": {
                    "observation.images.front": object(),
                    "observation.images.wrist": object(),
                    "observation.state": object(),
                },
            },
        )()

        def select_action(self, observation):
            return torch.tensor([[0.0, 0.25]])

    def fake_load_policy_and_processors(**kwargs):
        captured["stats_meta"] = kwargs["stats_meta"]
        return FakePolicy(), lambda batch: batch, lambda action: action

    monkeypatch.setattr(module, "LeRobotDataset", FakeDatasetForImagenetStats)
    monkeypatch.setattr(module, "load_policy_and_processors", fake_load_policy_and_processors)

    module.run_inspection(
        policy_path=Path("/tmp/pretrained_model"),
        dataset_repo_id="lekiwi_pick_and_put_banana",
        dataset_root=Path("/tmp/dataset"),
        output_dir=tmp_path,
        episode_index=0,
        frame_index=0,
    )

    front_stats = captured["stats_meta"].stats["observation.images.front"]
    wrist_stats = captured["stats_meta"].stats["observation.images.wrist"]

    assert torch.equal(front_stats["mean"], torch.tensor([0.485, 0.456, 0.406]))
    assert torch.equal(front_stats["std"], torch.tensor([0.229, 0.224, 0.225]))
    assert torch.equal(wrist_stats["mean"], torch.tensor([0.485, 0.456, 0.406]))
    assert torch.equal(wrist_stats["std"], torch.tensor([0.229, 0.224, 0.225]))


def test_run_inspection_uses_explicit_stats_dataset_source(tmp_path, monkeypatch):
    module = load_inspect_module()
    captured = {}

    class FakeDatasetForAlternateStats:
        def __init__(self, *args, **kwargs):
            self.meta = type(
                "Meta",
                (),
                {
                    "episodes": [{"dataset_from_index": 0, "dataset_to_index": 1}],
                    "stats": {"action": {"mean": torch.tensor([111.0, 222.0])}},
                    "features": {"action": {"names": ["joint", "theta.vel"]}},
                },
            )()
            self.num_frames = 1

        def __getitem__(self, idx):
            assert idx == 0
            return {
                "observation.images.front": torch.zeros(3, 4, 5),
                "observation.images.wrist": torch.ones(3, 4, 5),
                "observation.state": torch.tensor([1.0, 2.0, 3.0]),
                "action": torch.tensor([0.5, -0.25]),
                "episode_index": torch.tensor(0),
                "frame_index": torch.tensor(0),
                "index": torch.tensor(0),
                "task": "pick banana",
            }

    class FakeStatsMeta:
        def __init__(self, *, repo_id, root):
            self.repo_id = repo_id
            self.root = root
            self.camera_keys = []
            self.stats = {"action": {"mean": torch.tensor([9.0, 8.0])}}

    class FakePolicy:
        config = type(
            "Config",
            (),
            {
                "device": "cpu",
                "use_amp": False,
                "input_features": {
                    "observation.images.front": object(),
                    "observation.images.wrist": object(),
                    "observation.state": object(),
                },
            },
        )()

        def select_action(self, observation):
            return torch.tensor([[0.0, 0.25]])

    def fake_load_policy_and_processors(**kwargs):
        captured["dataset_meta"] = kwargs["dataset_meta"]
        captured["stats_meta"] = kwargs["stats_meta"]
        return FakePolicy(), lambda batch: batch, lambda action: action

    monkeypatch.setattr(module, "LeRobotDataset", FakeDatasetForAlternateStats)
    monkeypatch.setattr(module, "LeRobotDatasetMetadata", FakeStatsMeta)
    monkeypatch.setattr(module, "load_policy_and_processors", fake_load_policy_and_processors)

    result = module.run_inspection(
        policy_path=Path("pretrained_model"),
        dataset_repo_id="lekiwi_pick_and_put_banana",
        dataset_root=Path("dataset"),
        output_dir=tmp_path,
        global_index=0,
        stats_dataset_repo_id="training_stats_repo",
        stats_dataset_root=Path("training_stats_root"),
        use_imagenet_stats=False,
    )

    assert captured["stats_meta"] is not captured["dataset_meta"]
    assert captured["stats_meta"].repo_id == "training_stats_repo"
    assert captured["stats_meta"].root == Path("training_stats_root")
    assert torch.equal(captured["stats_meta"].stats["action"]["mean"], torch.tensor([9.0, 8.0]))
    assert torch.equal(captured["dataset_meta"].stats["action"]["mean"], torch.tensor([111.0, 222.0]))
    assert result["stats_dataset_repo_id"] == "training_stats_repo"
    assert result["stats_dataset_root"] == "training_stats_root"


def test_run_inspection_rejects_action_length_feature_name_mismatch(tmp_path, monkeypatch):
    module = load_inspect_module()

    class FakeDatasetForMismatch:
        def __init__(self, *args, **kwargs):
            self.meta = type(
                "Meta",
                (),
                {
                    "episodes": [{"dataset_from_index": 0, "dataset_to_index": 1}],
                    "stats": {"action": {"mean": torch.tensor([0.0, 0.0])}},
                    "features": {"action": {"names": ["joint", "theta.vel", "gripper"]}},
                },
            )()
            self.num_frames = 1

        def __getitem__(self, idx):
            return {
                "observation.images.front": torch.zeros(3, 4, 5),
                "observation.images.wrist": torch.ones(3, 4, 5),
                "observation.state": torch.tensor([1.0, 2.0, 3.0]),
                "action": torch.tensor([0.5, -0.25]),
                "episode_index": torch.tensor(0),
                "frame_index": torch.tensor(0),
                "index": torch.tensor(0),
                "task": "pick banana",
            }

    class FakePolicy:
        config = type(
            "Config",
            (),
            {
                "device": "cpu",
                "use_amp": False,
                "input_features": {
                    "observation.images.front": object(),
                    "observation.images.wrist": object(),
                    "observation.state": object(),
                },
            },
        )()

        def select_action(self, observation):
            return torch.tensor([[0.0, 0.25]])

    monkeypatch.setattr(module, "LeRobotDataset", FakeDatasetForMismatch)
    monkeypatch.setattr(
        module,
        "load_policy_and_processors",
        lambda **kwargs: (FakePolicy(), lambda batch: batch, lambda action: action),
    )

    with pytest.raises(ValueError, match="Action vector length 2 does not match action names length 3"):
        module.run_inspection(
            policy_path=Path("pretrained_model"),
            dataset_repo_id="lekiwi_pick_and_put_banana",
            dataset_root=Path("dataset"),
            output_dir=tmp_path,
            global_index=0,
        )


def test_parse_frame_index_arg_returns_half_open_range_bounds():
    module = load_inspect_module()
    mode, start, end = module.parse_frame_index_arg("2:5")

    assert mode == "range"
    assert start == 2
    assert end == 5


@pytest.mark.parametrize("frame_index", ["5:2", "3:3"])
def test_parse_frame_index_arg_rejects_reversed_and_empty_ranges(frame_index):
    module = load_inspect_module()

    with pytest.raises(ValueError, match="start < end"):
        module.parse_frame_index_arg(frame_index)


@pytest.mark.parametrize("episode_index", [-1, 2])
def test_resolve_sample_range_rejects_out_of_bounds_episode_index(episode_index):
    module = load_inspect_module()
    dataset = FakeDataset()

    with pytest.raises(IndexError, match=rf"episode_index {episode_index} is out of bounds"):
        module.resolve_sample_range(dataset=dataset, episode_index=episode_index, frame_index="0:2")


def test_main_accepts_frame_index_range_and_forwards_it_to_run_inspection(monkeypatch):
    module = load_inspect_module()
    captured = {}

    def fake_run_inspection(**kwargs):
        captured.update(kwargs)
        return {"status": "ok"}

    monkeypatch.setattr(module, "run_inspection", fake_run_inspection)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "inspect_frame_inference.py",
            "--policy-path",
            "pretrained_model",
            "--dataset-repo-id",
            "lekiwi_pick_and_put_banana",
            "--dataset-root",
            "dataset",
            "--output-dir",
            "outputs/inspect",
            "--episode-index",
            "1",
            "--frame-index",
            "1:3",
        ],
    )

    module.main()

    assert captured["episode_index"] == 1
    assert captured["frame_index"] == "1:3"


def test_main_accepts_no_reset_policy_and_forwards_it_to_run_inspection(monkeypatch):
    module = load_inspect_module()
    captured = {}

    def fake_run_inspection(**kwargs):
        captured.update(kwargs)
        return {"status": "ok"}

    monkeypatch.setattr(module, "run_inspection", fake_run_inspection)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "inspect_frame_inference.py",
            "--policy-path",
            "pretrained_model",
            "--dataset-repo-id",
            "lekiwi_pick_and_put_banana",
            "--dataset-root",
            "dataset",
            "--output-dir",
            "outputs/inspect",
            "--episode-index",
            "1",
            "--frame-index",
            "1:3",
            "--no-reset-policy",
        ],
    )

    module.main()

    assert captured["reset_policy"] is False


def test_save_range_artifacts_writes_clips_and_tabular_exports(tmp_path):
    module = load_inspect_module()
    rows = [
        {
            "global_index": 10,
            "episode_index": 1,
            "frame_index": 4,
            "prediction": {"theta.vel": 0.0},
            "label": {"theta.vel": 30.0},
            "delta": {"theta.vel": -30.0},
        },
        {
            "global_index": 11,
            "episode_index": 1,
            "frame_index": 5,
            "prediction": {"theta.vel": 30.0},
            "label": {"theta.vel": 30.0},
            "delta": {"theta.vel": 0.0},
        },
    ]
    samples = [
        {
            "observation.images.front": _make_rgb_frame((0.08, 0.71, 0.21)),
            "observation.images.wrist": _make_rgb_frame((0.84, 0.13, 0.58)),
        },
        {
            "observation.images.front": _make_rgb_frame((0.73, 0.19, 0.88)),
            "observation.images.wrist": _make_rgb_frame((0.16, 0.93, 0.27)),
        },
    ]

    module.save_range_artifacts(output_dir=tmp_path, samples=samples, comparison_rows=rows, fps=30)

    front_clip = tmp_path / "front_clip.mp4"
    wrist_clip = tmp_path / "wrist_clip.mp4"
    comparison_csv = tmp_path / "comparison.csv"
    comparison_json = tmp_path / "comparison.json"

    assert front_clip.exists()
    assert front_clip.stat().st_size > 0
    assert wrist_clip.exists()
    assert wrist_clip.stat().st_size > 0
    _assert_decodable_clip_matches_expected_order(front_clip, [(0.08, 0.71, 0.21), (0.73, 0.19, 0.88)])
    _assert_decodable_clip_matches_expected_order(wrist_clip, [(0.84, 0.13, 0.58), (0.16, 0.93, 0.27)])
    assert comparison_csv.exists()
    assert comparison_csv.stat().st_size > 0
    assert comparison_json.exists()
    assert comparison_json.stat().st_size > 0

    with comparison_csv.open(newline="", encoding="utf-8") as csv_file:
        csv_rows = list(csv.DictReader(csv_file))
    assert [row["global_index"] for row in csv_rows] == ["10", "11"]
    assert [row["episode_index"] for row in csv_rows] == ["1", "1"]
    assert [row["frame_index"] for row in csv_rows] == ["4", "5"]
    assert [row["prediction.theta.vel"] for row in csv_rows] == ["0.0", "30.0"]
    assert [row["label.theta.vel"] for row in csv_rows] == ["30.0", "30.0"]
    assert [row["delta.theta.vel"] for row in csv_rows] == ["-30.0", "0.0"]

    saved_summary = json.loads(comparison_json.read_text())
    assert saved_summary["episode_index"] == 1
    assert saved_summary["start"] == 4
    assert saved_summary["end"] == 6
    assert saved_summary["total_frames"] == 2
    assert saved_summary["action_names"] == ["theta.vel"]
    assert saved_summary["exported_paths"] == {
        "front_clip": str(front_clip),
        "wrist_clip": str(wrist_clip),
        "comparison_csv": str(comparison_csv),
        "comparison_json": str(comparison_json),
    }
    assert "rows" not in saved_summary


def test_run_inspection_range_returns_metadata_rows_and_writes_comparison_csv(tmp_path, monkeypatch):
    module = load_inspect_module()

    class FakeDatasetForRangeRun:
        def __init__(self, *args, **kwargs):
            self.meta = type(
                "Meta",
                (),
                {
                    "episodes": [
                        {"dataset_from_index": 0, "dataset_to_index": 2},
                        {"dataset_from_index": 2, "dataset_to_index": 6},
                    ],
                    "stats": {"action": {"mean": torch.tensor([0.0, 0.0])}},
                    "features": {"action": {"names": ["joint", "theta.vel"]}},
                },
            )()
            self.num_frames = 6

        def __getitem__(self, idx):
            episode_index = 0 if idx < 2 else 1
            frame_index = idx if idx < 2 else idx - 2
            if episode_index == 1:
                front_value, wrist_value = (
                    ((0.09, 0.72, 0.24), (0.82, 0.15, 0.61)),
                    ((0.74, 0.22, 0.86), (0.18, 0.91, 0.31)),
                )[frame_index - 1]
            else:
                front_value, wrist_value = (
                    ((0.03, 0.11, 0.19), (0.21, 0.07, 0.13)),
                    ((0.12, 0.05, 0.27), (0.06, 0.18, 0.09)),
                )[idx]
            return {
                "observation.images.front": _make_rgb_frame(front_value),
                "observation.images.wrist": _make_rgb_frame(wrist_value),
                "observation.state": torch.tensor([1.0, 2.0, 3.0]),
                "action": torch.tensor([0.5 + idx, -0.25 + idx]),
                "episode_index": torch.tensor(episode_index),
                "frame_index": torch.tensor(frame_index),
                "index": torch.tensor(idx),
                "task": "pick banana",
            }

    class FakePolicy:
        config = type(
            "Config",
            (),
            {
                "device": "cpu",
                "use_amp": False,
                "input_features": {
                    "observation.images.front": object(),
                    "observation.images.wrist": object(),
                    "observation.state": object(),
                },
            },
        )()

        def select_action(self, observation):
            batch_size = observation["observation.state"].shape[0]
            return torch.tensor([[0.0, 0.25]]).repeat(batch_size, 1)

    monkeypatch.setattr(module, "LeRobotDataset", FakeDatasetForRangeRun)
    monkeypatch.setattr(
        module,
        "load_policy_and_processors",
        lambda **kwargs: (FakePolicy(), lambda batch: batch, lambda action: action),
    )

    result = module.run_inspection(
        policy_path=Path("pretrained_model"),
        dataset_repo_id="lekiwi_pick_and_put_banana",
        dataset_root=Path("dataset"),
        output_dir=tmp_path,
        episode_index=1,
        frame_index="1:3",
    )

    assert result["policy_path"] == "pretrained_model"
    assert result["dataset_repo_id"] == "lekiwi_pick_and_put_banana"
    assert result["dataset_root"] == "dataset"
    assert result["stats_dataset_repo_id"] == "lekiwi_pick_and_put_banana"
    assert result["stats_dataset_root"] == "dataset"
    assert result["episode_index"] == 1
    assert result["range"] == {"start": 1, "end": 3}
    assert result["total_frames"] == 2
    assert "rows" not in result

    front_clip = tmp_path / "front_clip.mp4"
    wrist_clip = tmp_path / "wrist_clip.mp4"
    comparison_csv = tmp_path / "comparison.csv"
    comparison_json = tmp_path / "comparison.json"

    assert front_clip.exists()
    assert front_clip.stat().st_size > 0
    assert wrist_clip.exists()
    assert wrist_clip.stat().st_size > 0
    _assert_decodable_clip_matches_expected_order(front_clip, [(0.09, 0.72, 0.24), (0.74, 0.22, 0.86)])
    _assert_decodable_clip_matches_expected_order(wrist_clip, [(0.82, 0.15, 0.61), (0.18, 0.91, 0.31)])
    assert comparison_csv.exists()
    assert comparison_csv.stat().st_size > 0
    assert comparison_json.exists()
    assert comparison_json.stat().st_size > 0

    with comparison_csv.open(newline="", encoding="utf-8") as csv_file:
        csv_rows = list(csv.DictReader(csv_file))
    assert [row["episode_index"] for row in csv_rows] == ["1", "1"]
    assert [row["frame_index"] for row in csv_rows] == ["1", "2"]
    assert [row["global_index"] for row in csv_rows] == ["3", "4"]
    assert [row["prediction.joint"] for row in csv_rows] == ["0.0", "0.0"]
    assert [row["prediction.theta.vel"] for row in csv_rows] == ["0.25", "0.25"]
    assert [row["label.joint"] for row in csv_rows] == ["3.5", "4.5"]
    assert [row["label.theta.vel"] for row in csv_rows] == ["2.75", "3.75"]
    assert [row["delta.joint"] for row in csv_rows] == ["-3.5", "-4.5"]
    assert [row["delta.theta.vel"] for row in csv_rows] == ["-2.5", "-3.5"]

    saved_summary = json.loads(comparison_json.read_text())
    assert saved_summary["policy_path"] == "pretrained_model"
    assert saved_summary["dataset_repo_id"] == "lekiwi_pick_and_put_banana"
    assert saved_summary["dataset_root"] == "dataset"
    assert saved_summary["stats_dataset_repo_id"] == "lekiwi_pick_and_put_banana"
    assert saved_summary["stats_dataset_root"] == "dataset"
    assert saved_summary["episode_index"] == 1
    assert saved_summary["start"] == 1
    assert saved_summary["end"] == 3
    assert saved_summary["total_frames"] == 2
    assert saved_summary["action_names"] == ["joint", "theta.vel"]
    assert saved_summary["exported_paths"] == {
        "front_clip": str(front_clip),
        "wrist_clip": str(wrist_clip),
        "comparison_csv": str(comparison_csv),
        "comparison_json": str(comparison_json),
    }
    assert "rows" not in saved_summary


def test_run_inspection_range_rejects_out_of_bounds_frame_range(tmp_path, monkeypatch):
    module = load_inspect_module()

    class FakeDatasetForRangeBounds:
        def __init__(self, *args, **kwargs):
            self.meta = type(
                "Meta",
                (),
                {
                    "episodes": [
                        {"dataset_from_index": 0, "dataset_to_index": 2},
                        {"dataset_from_index": 2, "dataset_to_index": 6},
                    ],
                    "stats": {"action": {"mean": torch.tensor([0.0, 0.0])}},
                    "features": {"action": {"names": ["joint", "theta.vel"]}},
                },
            )()
            self.num_frames = 6

        def __getitem__(self, idx):
            raise AssertionError("Out-of-bounds frame ranges should fail before dataset access.")

    class FakePolicy:
        config = type(
            "Config",
            (),
            {
                "device": "cpu",
                "use_amp": False,
                "input_features": {
                    "observation.images.front": object(),
                    "observation.images.wrist": object(),
                    "observation.state": object(),
                },
            },
        )()

        def select_action(self, observation):
            return torch.tensor([[0.0, 0.25]])

    monkeypatch.setattr(module, "LeRobotDataset", FakeDatasetForRangeBounds)
    monkeypatch.setattr(
        module,
        "load_policy_and_processors",
        lambda **kwargs: (FakePolicy(), lambda batch: batch, lambda action: action),
    )

    with pytest.raises(IndexError, match="out of bounds"):
        module.run_inspection(
            policy_path=Path("pretrained_model"),
            dataset_repo_id="lekiwi_pick_and_put_banana",
            dataset_root=Path("dataset"),
            output_dir=tmp_path,
            episode_index=1,
            frame_index="1:5",
        )


def test_run_inspection_range_resets_policy_before_each_frame(tmp_path, monkeypatch):
    module = load_inspect_module()

    class FakeDatasetForStatefulRange:
        def __init__(self, *args, **kwargs):
            self.meta = type(
                "Meta",
                (),
                {
                    "episodes": [{"dataset_from_index": 0, "dataset_to_index": 3}],
                    "stats": {"action": {"mean": torch.tensor([0.0])}},
                    "features": {"action": {"names": ["joint"]}},
                },
            )()
            self.num_frames = 3

        def __getitem__(self, idx):
            return {
                "observation.images.front": _make_rgb_frame((0.1, 0.2, 0.3)),
                "observation.images.wrist": _make_rgb_frame((0.4, 0.5, 0.6)),
                "observation.state": torch.tensor([1.0, 2.0, 3.0]),
                "action": torch.tensor([float(idx)]),
                "episode_index": torch.tensor(0),
                "frame_index": torch.tensor(idx),
                "index": torch.tensor(idx),
                "task": "pick banana",
            }

    class StatefulFakePolicy:
        config = type(
            "Config",
            (),
            {
                "device": "cpu",
                "use_amp": False,
                "input_features": {
                    "observation.images.front": object(),
                    "observation.images.wrist": object(),
                    "observation.state": object(),
                },
            },
        )()

        def __init__(self):
            self.counter = 0
            self.reset_calls = 0

        def reset(self):
            self.counter = 0
            self.reset_calls += 1

        def select_action(self, observation):
            value = float(self.counter)
            self.counter += 1
            return torch.tensor([[value]])

    policy = StatefulFakePolicy()
    monkeypatch.setattr(module, "LeRobotDataset", FakeDatasetForStatefulRange)
    monkeypatch.setattr(
        module,
        "load_policy_and_processors",
        lambda **kwargs: (policy, lambda batch: batch, lambda action: action),
    )

    result = module.run_inspection(
        policy_path=Path("pretrained_model"),
        dataset_repo_id="lekiwi_pick_and_put_banana",
        dataset_root=Path("dataset"),
        output_dir=tmp_path,
        episode_index=0,
        frame_index="0:3",
    )

    assert "rows" not in result
    assert policy.reset_calls == 3


def test_run_inspection_range_skips_reset_when_disabled(tmp_path, monkeypatch):
    module = load_inspect_module()

    class FakeDatasetForStatefulRange:
        def __init__(self, *args, **kwargs):
            self.meta = type(
                "Meta",
                (),
                {
                    "episodes": [{"dataset_from_index": 0, "dataset_to_index": 3}],
                    "stats": {"action": {"mean": torch.tensor([0.0])}},
                    "features": {"action": {"names": ["joint"]}},
                },
            )()
            self.num_frames = 3

        def __getitem__(self, idx):
            return {
                "observation.images.front": _make_rgb_frame((0.1, 0.2, 0.3)),
                "observation.images.wrist": _make_rgb_frame((0.4, 0.5, 0.6)),
                "observation.state": torch.tensor([1.0, 2.0, 3.0]),
                "action": torch.tensor([float(idx)]),
                "episode_index": torch.tensor(0),
                "frame_index": torch.tensor(idx),
                "index": torch.tensor(idx),
                "task": "pick banana",
            }

    class StatefulFakePolicy:
        config = type(
            "Config",
            (),
            {
                "device": "cpu",
                "use_amp": False,
                "input_features": {
                    "observation.images.front": object(),
                    "observation.images.wrist": object(),
                    "observation.state": object(),
                },
            },
        )()

        def __init__(self):
            self.reset_calls = 0

        def reset(self):
            self.reset_calls += 1

        def select_action(self, observation):
            return torch.tensor([[0.0]])

    policy = StatefulFakePolicy()
    monkeypatch.setattr(module, "LeRobotDataset", FakeDatasetForStatefulRange)
    monkeypatch.setattr(
        module,
        "load_policy_and_processors",
        lambda **kwargs: (policy, lambda batch: batch, lambda action: action),
    )

    module.run_inspection(
        policy_path=Path("pretrained_model"),
        dataset_repo_id="lekiwi_pick_and_put_banana",
        dataset_root=Path("dataset"),
        output_dir=tmp_path,
        episode_index=0,
        frame_index="0:3",
        reset_policy=False,
    )

    assert policy.reset_calls == 0
