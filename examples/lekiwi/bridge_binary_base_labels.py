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

"""Analyze and bridge short inactive gaps in LeKiwi base-control labels.

Example:
    python examples/lekiwi/bridge_binary_base_labels.py \
        --src-root /mnt/data/yzh/dataset/lekiwi_pick_and_put_banana \
        --dst-root /mnt/data/yzh/dataset/lekiwi_pick_and_put_banana_gap3 \
        --gap-threshold 3 \
        --process-indices 8 \
        --target-values 30
"""

import argparse
import logging
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from lerobot.datasets.dataset_tools import recompute_stats
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def bridge_target_value_gaps(
    values: np.ndarray,
    gap_threshold: int,
    *,
    target_value: float,
    inactive_value: float = 0.0,
    atol: float = 1e-6,
) -> np.ndarray:
    if gap_threshold < 0:
        raise ValueError(f"gap_threshold must be >= 0, got {gap_threshold}")

    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 1:
        raise ValueError(f"Expected a 1D control array, got shape {values.shape}")

    bridged_values = values.copy()
    is_target = np.isclose(values, target_value, atol=atol)
    changes = np.diff(np.r_[False, is_target, False].astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1) - 1

    for prev_end, next_start in zip(ends[:-1], starts[1:]):
        gap = next_start - prev_end - 1
        if gap <= 0 or gap > gap_threshold:
            continue
        gap_values = values[prev_end + 1 : next_start]
        if np.all(np.isclose(gap_values, inactive_value, atol=atol)):
            bridged_values[prev_end + 1 : next_start] = target_value

    return bridged_values.astype(np.float32)


def bridge_binary_theta_velocity(
    theta_values: np.ndarray,
    gap_threshold: int,
    *,
    active_value: float = 30.0,
    inactive_value: float = 0.0,
    atol: float = 1e-6,
) -> np.ndarray:
    if gap_threshold < 0:
        raise ValueError(f"gap_threshold must be >= 0, got {gap_threshold}")

    values = np.asarray(theta_values, dtype=np.float32)
    if values.ndim != 1:
        raise ValueError(f"Expected a 1D theta array, got shape {values.shape}")

    is_active = np.isclose(values, active_value, atol=atol)
    is_inactive = np.isclose(values, inactive_value, atol=atol)
    if not np.all(is_active | is_inactive):
        invalid_values = np.unique(values[~(is_active | is_inactive)])
        raise ValueError(
            f"Expected only binary theta values {{{inactive_value}, {active_value}}}, got {invalid_values.tolist()}"
        )

    return bridge_target_value_gaps(
        values,
        gap_threshold,
        target_value=active_value,
        inactive_value=inactive_value,
        atol=atol,
    )


def _rounded_float(value: float, decimals: int = 6) -> float:
    return float(np.round(float(value), decimals))


def _count_distribution(values: np.ndarray) -> dict[float, int]:
    rounded_values = np.round(np.asarray(values, dtype=np.float32), 6)
    unique_values, counts = np.unique(rounded_values, return_counts=True)
    return {_rounded_float(value): int(count) for value, count in zip(unique_values, counts, strict=True)}


def _compute_gap_lengths(
    values: np.ndarray,
    *,
    target_value: float,
    inactive_value: float = 0.0,
    atol: float = 1e-6,
) -> list[int]:
    values = np.asarray(values, dtype=np.float32)
    is_target = np.isclose(values, target_value, atol=atol)
    changes = np.diff(np.r_[False, is_target, False].astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1) - 1

    gap_lengths: list[int] = []
    for prev_end, next_start in zip(ends[:-1], starts[1:]):
        gap = next_start - prev_end - 1
        if gap <= 0:
            continue
        gap_values = values[prev_end + 1 : next_start]
        if np.all(np.isclose(gap_values, inactive_value, atol=atol)):
            gap_lengths.append(int(gap))
    return gap_lengths


def analyze_action_array(
    actions: np.ndarray,
    episode_index: np.ndarray,
    *,
    analyze_indices: list[int],
    inactive_value: float = 0.0,
    atol: float = 1e-6,
) -> dict[int, dict[str, dict]]:
    if actions.ndim != 2:
        raise ValueError(f"Expected a 2D action array, got shape {actions.shape}")

    analysis: dict[int, dict[str, dict]] = {}
    unique_episode_indices = np.unique(episode_index)
    for action_index in analyze_indices:
        if action_index >= actions.shape[1]:
            raise IndexError(f"action index {action_index} is out of range for action dim {actions.shape[1]}")

        index_values = actions[:, action_index]
        value_counts = _count_distribution(index_values)
        gap_counts: dict[float, dict[int, int]] = {}
        for target_value in value_counts:
            if np.isclose(target_value, inactive_value, atol=atol):
                continue
            gap_counter: Counter[int] = Counter()
            for ep_idx in unique_episode_indices:
                episode_values = index_values[episode_index == ep_idx]
                gap_counter.update(
                    _compute_gap_lengths(
                        episode_values,
                        target_value=target_value,
                        inactive_value=inactive_value,
                        atol=atol,
                    )
                )
            gap_counts[target_value] = dict(sorted(gap_counter.items()))

        analysis[action_index] = {
            "value_counts": dict(sorted(value_counts.items())),
            "gap_counts": dict(sorted(gap_counts.items())),
        }
    return analysis


def analyze_action_parquet_file(
    parquet_path: Path,
    *,
    analyze_indices: list[int],
    action_column: str = "action",
    episode_index_column: str = "episode_index",
    inactive_value: float = 0.0,
    atol: float = 1e-6,
) -> dict[int, dict[str, dict]]:
    table = pq.read_table(parquet_path)
    actions = np.asarray(table[action_column].to_pylist(), dtype=np.float32)
    episode_index = np.asarray(table[episode_index_column].to_pylist(), dtype=np.int64)
    return analyze_action_array(
        actions,
        episode_index,
        analyze_indices=analyze_indices,
        inactive_value=inactive_value,
        atol=atol,
    )


def _merge_analysis(
    aggregate: dict[int, dict[str, dict]],
    partial: dict[int, dict[str, dict]],
) -> dict[int, dict[str, dict]]:
    for action_index, stats in partial.items():
        aggregate.setdefault(action_index, {"value_counts": {}, "gap_counts": {}})
        for value, count in stats["value_counts"].items():
            aggregate[action_index]["value_counts"][value] = aggregate[action_index]["value_counts"].get(value, 0) + count
        for target_value, gap_counts in stats["gap_counts"].items():
            target_gap_counts = aggregate[action_index]["gap_counts"].setdefault(target_value, {})
            for gap, count in gap_counts.items():
                target_gap_counts[gap] = target_gap_counts.get(gap, 0) + count
    return aggregate


def analyze_dataset(
    dataset_root: Path,
    *,
    analyze_indices: list[int],
    inactive_value: float = 0.0,
    atol: float = 1e-6,
) -> dict[int, dict[str, dict]]:
    parquet_files = sorted((dataset_root / "data").glob("*/*.parquet"))
    if not parquet_files:
        raise ValueError(f"No parquet files found under {dataset_root / 'data'}")

    aggregate: dict[int, dict[str, dict]] = {}
    for parquet_path in parquet_files:
        partial = analyze_action_parquet_file(
            parquet_path,
            analyze_indices=analyze_indices,
            inactive_value=inactive_value,
            atol=atol,
        )
        _merge_analysis(aggregate, partial)

    for action_index, stats in aggregate.items():
        stats["value_counts"] = dict(sorted(stats["value_counts"].items()))
        stats["gap_counts"] = {
            target_value: dict(sorted(gap_counts.items()))
            for target_value, gap_counts in sorted(stats["gap_counts"].items())
        }
    return aggregate


def format_analysis_summary(analysis: dict[int, dict[str, dict]]) -> str:
    lines: list[str] = []
    for action_index, stats in sorted(analysis.items()):
        lines.append(f"action[{action_index}] value counts:")
        for value, count in stats["value_counts"].items():
            lines.append(f"  {value}: {count}")
        if stats["gap_counts"]:
            lines.append(f"action[{action_index}] gap counts:")
            for target_value, gap_counts in stats["gap_counts"].items():
                if gap_counts:
                    gap_summary = ", ".join(f"{gap}:{count}" for gap, count in gap_counts.items())
                else:
                    gap_summary = "none"
                lines.append(f"  target {target_value}: {gap_summary}")
        else:
            lines.append(f"action[{action_index}] gap counts: none")
    return "\n".join(lines)


def _resolve_processing_targets(
    *,
    indices: list[int] | None,
    target_values: list[float] | None,
    theta_index: int,
    active_value: float,
) -> tuple[list[int], list[float]]:
    if indices is None and target_values is None:
        return [theta_index], [active_value]
    if indices is None or target_values is None:
        raise ValueError("`indices` and `target_values` must be provided together.")
    if len(indices) != len(target_values):
        raise ValueError(
            f"`indices` and `target_values` must have the same length, got {len(indices)} and {len(target_values)}."
        )
    return indices, target_values


def rewrite_action_parquet_file(
    parquet_path: Path,
    *,
    gap_threshold: int,
    indices: list[int] | None = None,
    target_values: list[float] | None = None,
    theta_index: int = 8,
    action_column: str = "action",
    episode_index_column: str = "episode_index",
    active_value: float = 30.0,
    inactive_value: float = 0.0,
) -> None:
    table = pq.read_table(parquet_path)
    schema = table.schema

    action_field_index = schema.get_field_index(action_column)
    if action_field_index < 0:
        raise KeyError(f"Column '{action_column}' not found in {parquet_path}")

    episode_field_index = schema.get_field_index(episode_index_column)
    if episode_field_index < 0:
        raise KeyError(f"Column '{episode_index_column}' not found in {parquet_path}")

    actions = np.asarray(table[action_column].to_pylist(), dtype=np.float32)
    if actions.ndim != 2:
        raise ValueError(f"Expected a 2D action array, got shape {actions.shape} in {parquet_path}")

    process_indices, process_target_values = _resolve_processing_targets(
        indices=indices,
        target_values=target_values,
        theta_index=theta_index,
        active_value=active_value,
    )
    for action_index in process_indices:
        if action_index >= actions.shape[1]:
            raise IndexError(f"action index {action_index} is out of range for action dim {actions.shape[1]}")

    episode_index = np.asarray(table[episode_index_column].to_pylist(), dtype=np.int64)
    updated_actions = actions.copy()

    for ep_idx in np.unique(episode_index):
        episode_mask = episode_index == ep_idx
        for action_index, target_value in zip(process_indices, process_target_values, strict=True):
            updated_actions[episode_mask, action_index] = bridge_target_value_gaps(
                updated_actions[episode_mask, action_index],
                gap_threshold,
                target_value=target_value,
                inactive_value=inactive_value,
            )

    updated_action_array = pa.array(updated_actions.tolist(), type=schema.field(action_column).type)
    updated_table = table.set_column(action_field_index, schema.field(action_column), updated_action_array)
    pq.write_table(updated_table, parquet_path, compression="snappy")


def process_dataset(
    src_root: Path,
    dst_root: Path,
    *,
    gap_threshold: int,
    indices: list[int] | None = None,
    target_values: list[float] | None = None,
    theta_index: int = 8,
    repo_id: str | None = None,
    active_value: float = 30.0,
    inactive_value: float = 0.0,
    skip_recompute_stats: bool = False,
) -> None:
    if not src_root.exists():
        raise FileNotFoundError(f"Source dataset does not exist: {src_root}")
    if dst_root.exists():
        raise FileExistsError(f"Destination already exists: {dst_root}")

    logging.info("Copying dataset from %s to %s", src_root, dst_root)
    shutil.copytree(src_root, dst_root)

    parquet_files = sorted((dst_root / "data").glob("*/*.parquet"))
    if not parquet_files:
        raise ValueError(f"No parquet files found under {dst_root / 'data'}")

    for parquet_path in parquet_files:
        logging.info("Rewriting %s", parquet_path)
        rewrite_action_parquet_file(
            parquet_path,
            gap_threshold=gap_threshold,
            indices=indices,
            target_values=target_values,
            theta_index=theta_index,
            active_value=active_value,
            inactive_value=inactive_value,
        )

    if skip_recompute_stats:
        logging.info("Skipping stats recomputation")
        return

    dataset = LeRobotDataset(repo_id=repo_id or dst_root.name, root=dst_root)
    recompute_stats(dataset, skip_image_video=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src-root", type=Path, required=True, help="Source dataset directory")
    parser.add_argument("--dst-root", type=Path, default=None, help="Destination dataset directory for rewritten data")
    parser.add_argument("--gap-threshold", type=int, default=None, help="Fill inactive gaps up to this many frames")
    parser.add_argument(
        "--analyze-indices",
        type=int,
        nargs="+",
        default=[6, 7, 8],
        help="Action indices to analyze before selecting a gap threshold",
    )
    parser.add_argument(
        "--process-indices",
        type=int,
        nargs="+",
        default=None,
        help="Action indices to bridge when rewriting the dataset",
    )
    parser.add_argument(
        "--target-values",
        type=float,
        nargs="+",
        default=None,
        help="Target values to bridge for each process index",
    )
    parser.add_argument("--theta-index", type=int, default=8, help="Theta velocity index inside the action vector")
    parser.add_argument("--active-value", type=float, default=30.0, help="Legacy single-target bridge value")
    parser.add_argument("--inactive-value", type=float, default=0.0, help="Inactive value inside a bridged gap")
    parser.add_argument("--repo-id", type=str, default=None, help="Optional repo id to use when recomputing stats")
    parser.add_argument(
        "--skip-recompute-stats",
        action="store_true",
        help="Skip rewriting meta/stats.json after modifying action labels",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    analysis = analyze_dataset(
        args.src_root,
        analyze_indices=args.analyze_indices,
        inactive_value=args.inactive_value,
    )
    logging.info("Control analysis for %s\n%s", args.src_root, format_analysis_summary(analysis))

    if args.gap_threshold is None:
        logging.info("Analysis complete. Re-run with --gap-threshold, --dst-root, and processing arguments to rewrite.")
        return
    if args.dst_root is None:
        raise ValueError("`--dst-root` is required when `--gap-threshold` is provided.")
    process_dataset(
        args.src_root,
        args.dst_root,
        gap_threshold=args.gap_threshold,
        indices=args.process_indices,
        target_values=args.target_values,
        theta_index=args.theta_index,
        repo_id=args.repo_id,
        active_value=args.active_value,
        inactive_value=args.inactive_value,
        skip_recompute_stats=args.skip_recompute_stats,
    )


if __name__ == "__main__":
    main()
