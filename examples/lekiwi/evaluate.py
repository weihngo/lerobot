# !/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

import math
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.factory import IMAGENET_STATS
from lerobot.datasets.feature_utils import hw_to_dataset_features
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.processor import make_default_processors
from lerobot.robots.lekiwi import LeKiwiClient, LeKiwiClientConfig
from lerobot.scripts.lerobot_record import record_loop
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun

LEKIWI_BASE_ACTION_KEYS = ("x.vel", "y.vel", "theta.vel")
NUM_EPISODES = 2
FPS = 30
EPISODE_TIME_SEC = 180
TASK_DESCRIPTION = "pick and put banana"
TASK_DESCRIPTION_FILE = Path("/tmp/lekiwi_task.txt")
HF_MODEL_ID = "/home/lwh/code/lerobot/outputs_hdd/find_and_walk_banana_gap10/checkpoints/last/pretrained_model"
HF_DATASET_ID = "lwh/pi05"
TRAIN_STATS_DATASET_ID: str | None = None
TRAIN_STATS_DATASET_ROOT: Path | None = None

TRAIN_STATS_DATASET_ID = "lekiwi_pick_and_put_banana"
TRAIN_STATS_DATASET_ROOT = Path("/mnt/data/yzh/dataset/lekiwi_pick_and_put_banana")


def resolve_job_name_from_pretrained_path(pretrained_path: str | Path) -> str:
    path = Path(pretrained_path)
    if "checkpoints" in path.parts:
        checkpoint_index = path.parts.index("checkpoints")
        if checkpoint_index > 0:
            return path.parts[checkpoint_index - 1]
    return path.name or path.stem


def resolve_control_log_path(pretrained_path: str | Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"lekiwi_control_{timestamp}.txt"
    return Path("logs") / resolve_job_name_from_pretrained_path(pretrained_path) / filename


def append_control_log(log_path: Path, action: dict[str, float]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().isoformat(timespec="milliseconds")
    action_str = " ".join(f"{key}={value}" for key, value in sorted(action.items()))
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(f"{timestamp} {action_str}\n")


class FileTaskProvider:
    def __init__(self, path: str | Path, *, default_task: str):
        self.path = Path(path)
        self._task = default_task.strip()
        if not self._task:
            raise ValueError("default_task must not be empty")
        self._mtime_ns: int | None = None
        self._ensure_file()
        self._refresh(force=True)

    def __call__(self) -> str:
        self._refresh()
        return self._task

    def _ensure_file(self) -> None:
        if self.path.exists():
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(f"{self._task}\n", encoding="utf-8")

    def _refresh(self, *, force: bool = False) -> None:
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            self._ensure_file()
            stat = self.path.stat()

        if not force and stat.st_mtime_ns == self._mtime_ns:
            return

        self._mtime_ns = stat.st_mtime_ns
        task = self.path.read_text(encoding="utf-8").strip()
        if not task:
            return
        if task != self._task:
            print(f"Task description updated: {task}")
        self._task = task


def resolve_policy_stats(
    *,
    eval_dataset_stats,
    train_dataset_repo_id: str | None,
    train_dataset_root: Path | None,
):
    if train_dataset_repo_id is None:
        return eval_dataset_stats

    train_meta = LeRobotDatasetMetadata(repo_id=train_dataset_repo_id, root=train_dataset_root)
    train_meta = deepcopy(train_meta)
    for camera_key in getattr(train_meta, "camera_keys", []):
        if camera_key not in train_meta.stats:
            continue
        for stats_type, stats in IMAGENET_STATS.items():
            train_meta.stats[camera_key][stats_type] = torch.tensor(stats, dtype=torch.float32)
    return train_meta.stats


def load_policy_for_evaluate(*, pretrained_path: str | Path, dataset_stats):
    config = PreTrainedConfig.from_pretrained(pretrained_path)
    policy_class = get_policy_class(config.type)
    return policy_class.from_pretrained(pretrained_path, config=config, dataset_stats=dataset_stats)


def build_lekiwi_base_stop_action(action: dict[str, float]) -> dict[str, float]:
    stop_action = dict(action)
    for key in LEKIWI_BASE_ACTION_KEYS:
        if key in stop_action:
            stop_action[key] = 0.0
    return stop_action


def sanitize_lekiwi_base_action(action: dict[str, float]) -> dict[str, float]:
    sanitized_action = dict(action)
    for key in LEKIWI_BASE_ACTION_KEYS:
        if key in sanitized_action:
            value = float(sanitized_action[key])
            sanitized_action[key] = value if math.isfinite(value) else 0.0
    return sanitized_action


def main():
    # Create the robot configuration & robot
    robot_config = LeKiwiClientConfig(remote_ip="192.168.10.102", id="lekiwi")

    robot = LeKiwiClient(robot_config)

    # Configure the dataset features
    action_features = hw_to_dataset_features(robot.action_features, ACTION)
    obs_features = hw_to_dataset_features(robot.observation_features, OBS_STR)
    dataset_features = {**action_features, **obs_features}

    # Create the dataset
    dataset = LeRobotDataset.create(
        repo_id=HF_DATASET_ID,
        fps=FPS,
        features=dataset_features,
        robot_type=robot.name,
        use_videos=True,
        image_writer_threads=4,
    )

    policy_stats = resolve_policy_stats(
        eval_dataset_stats=dataset.meta.stats,
        train_dataset_repo_id=TRAIN_STATS_DATASET_ID,
        train_dataset_root=TRAIN_STATS_DATASET_ROOT,
    )

    # Create policy
    policy = load_policy_for_evaluate(
        pretrained_path=HF_MODEL_ID,
        dataset_stats=policy_stats,
    )

    # Build Policy Processors
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=HF_MODEL_ID,
        dataset_stats=policy_stats,
        # The inference device is automatically set to match the detected hardware, overriding any previous device settings from training to ensure compatibility.
        preprocessor_overrides={"device_processor": {"device": str(policy.config.device)}},
    )

    # Connect the robot
    # To connect you already should have this script running on LeKiwi: `python -m lerobot.robots.lekiwi.lekiwi_host --robot.id=my_awesome_kiwi`
    robot.connect()

    # TODO(Steven): Update this example to use pipelines
    teleop_action_processor, base_robot_action_processor, robot_observation_processor = make_default_processors()
    control_log_path = resolve_control_log_path(HF_MODEL_ID)

    def robot_action_processor(action_and_observation):
        robot_action = base_robot_action_processor(action_and_observation)
        robot_action = sanitize_lekiwi_base_action(robot_action)
        append_control_log(control_log_path, robot_action)
        return robot_action

    def stop_action_builder(action: dict[str, float], observation):
        return robot_action_processor((build_lekiwi_base_stop_action(action), observation))

    # Initialize the keyboard listener and rerun visualization
    listener, events = init_keyboard_listener()
    init_rerun(session_name="lekiwi_evaluate")
    task_provider = FileTaskProvider(TASK_DESCRIPTION_FILE, default_task=TASK_DESCRIPTION)
    print(f"Task description file: {TASK_DESCRIPTION_FILE}")

    try:
        if not robot.is_connected:
            raise ValueError("Robot is not connected!")

        print("Starting evaluate loop...")
        recorded_episodes = 0
        while recorded_episodes < NUM_EPISODES and not events["stop_recording"]:
            log_say(f"Running inference, recording eval episode {recorded_episodes} of {NUM_EPISODES}")

            # Main record loop
            record_loop(
                robot=robot,
                events=events,
                fps=FPS,
                policy=policy,
                preprocessor=preprocessor,  # Pass the pre and post policy processors
                postprocessor=postprocessor,
                dataset=dataset,
                control_time_s=EPISODE_TIME_SEC,
                single_task=TASK_DESCRIPTION,
                task_provider=task_provider,
                display_data=True,
                teleop_action_processor=teleop_action_processor,
                robot_action_processor=robot_action_processor,
                robot_observation_processor=robot_observation_processor,
                stop_action_builder=stop_action_builder,
            )

            # Reset the environment if not stopping or re-recording
            if not events["stop_recording"] and (
                (recorded_episodes < NUM_EPISODES - 1) or events["rerecord_episode"]
            ):
                log_say("Reset the environment")
                record_loop(
                    robot=robot,
                    events=events,
                    fps=FPS,
                    control_time_s=EPISODE_TIME_SEC,
                    single_task=TASK_DESCRIPTION,
                    task_provider=task_provider,
                    display_data=True,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    stop_action_builder=stop_action_builder,
                )

            if events["rerecord_episode"]:
                log_say("Re-record episode")
                events["rerecord_episode"] = False
                events["exit_early"] = False
                dataset.clear_episode_buffer()
                continue

            # Save episode
            dataset.save_episode()
            recorded_episodes += 1

    finally:
        # Clean up
        log_say("Stop recording")
        robot.disconnect()
        listener.stop()

        dataset.finalize()
        # dataset.push_to_hub()


if __name__ == "__main__":
    main()
