#!/usr/bin/env python

import importlib.util
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


def load_bridge_module():
    module_path = Path("/home/lwh/code/lerobot/examples/lekiwi/bridge_binary_base_labels.py")
    spec = importlib.util.spec_from_file_location("examples.lekiwi.bridge_binary_base_labels", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_bridge_theta_velocity_fills_gaps_up_to_threshold():
    module = load_bridge_module()

    theta = np.array([0.0, 30.0, 30.0, 0.0, 0.0, 30.0, 0.0, 0.0, 0.0, 30.0, 0.0], dtype=np.float32)

    bridged = module.bridge_binary_theta_velocity(theta, gap_threshold=2)

    np.testing.assert_array_equal(
        bridged,
        np.array([0.0, 30.0, 30.0, 30.0, 30.0, 30.0, 0.0, 0.0, 0.0, 30.0, 0.0], dtype=np.float32),
    )


def test_bridge_theta_velocity_rejects_non_binary_values():
    module = load_bridge_module()

    theta = np.array([0.0, 30.0, 15.0, 0.0], dtype=np.float32)

    with pytest.raises(ValueError, match="Expected only binary theta values"):
        module.bridge_binary_theta_velocity(theta, gap_threshold=2)


def test_rewrite_action_parquet_file_updates_theta_per_episode(tmp_path):
    module = load_bridge_module()

    actions = [
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 20.0, 30.0],
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 20.0, 0.0],
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 20.0, 30.0],
        [7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 40.0, 50.0, 30.0],
        [7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 40.0, 50.0, 0.0],
        [7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 40.0, 50.0, 0.0],
        [7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 40.0, 50.0, 30.0],
    ]
    episode_index = [0, 0, 0, 1, 1, 1, 1]
    frame_index = [0, 1, 2, 0, 1, 2, 3]
    schema = pa.schema(
        [
            ("action", pa.list_(pa.float32(), 9)),
            ("episode_index", pa.int64()),
            ("frame_index", pa.int64()),
        ]
    )
    table = pa.Table.from_arrays(
        [
            pa.array(actions, type=schema.field("action").type),
            pa.array(episode_index, type=pa.int64()),
            pa.array(frame_index, type=pa.int64()),
        ],
        schema=schema,
    )
    parquet_path = tmp_path / "actions.parquet"
    pq.write_table(table, parquet_path)

    module.rewrite_action_parquet_file(parquet_path, gap_threshold=1)

    rewritten = pq.read_table(parquet_path)
    rewritten_actions = np.array(rewritten["action"].to_pylist(), dtype=np.float32)

    np.testing.assert_array_equal(rewritten_actions[:, :8], np.array(actions, dtype=np.float32)[:, :8])
    np.testing.assert_array_equal(rewritten_actions[:3, 8], np.array([30.0, 30.0, 30.0], dtype=np.float32))
    np.testing.assert_array_equal(rewritten_actions[3:, 8], np.array([30.0, 0.0, 0.0, 30.0], dtype=np.float32))


def test_analyze_action_parquet_file_reports_value_and_gap_distributions(tmp_path):
    module = load_bridge_module()

    actions = [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 30.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 30.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.1, 0.1, -30.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.1, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.1, 0.1, -30.0],
    ]
    episode_index = [0, 0, 0, 1, 1, 1]
    schema = pa.schema(
        [
            ("action", pa.list_(pa.float32(), 9)),
            ("episode_index", pa.int64()),
        ]
    )
    table = pa.Table.from_arrays(
        [
            pa.array(actions, type=schema.field("action").type),
            pa.array(episode_index, type=pa.int64()),
        ],
        schema=schema,
    )
    parquet_path = tmp_path / "actions.parquet"
    pq.write_table(table, parquet_path)

    analysis = module.analyze_action_parquet_file(parquet_path, analyze_indices=[6, 7, 8], inactive_value=0.0)

    assert analysis[6]["value_counts"] == {-0.1: 3, 0.0: 1, 0.1: 2}
    assert analysis[7]["value_counts"] == {0.0: 4, 0.1: 2}
    assert analysis[8]["value_counts"] == {-30.0: 2, 0.0: 2, 30.0: 2}
    assert analysis[6]["gap_counts"][0.1] == {1: 1}
    assert analysis[8]["gap_counts"][30.0] == {1: 1}
    assert analysis[8]["gap_counts"][-30.0] == {1: 1}


def test_rewrite_action_parquet_file_updates_multiple_control_indices(tmp_path):
    module = load_bridge_module()

    actions = [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 30.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 30.0],
    ]
    episode_index = [0, 0, 0]
    schema = pa.schema(
        [
            ("action", pa.list_(pa.float32(), 9)),
            ("episode_index", pa.int64()),
        ]
    )
    table = pa.Table.from_arrays(
        [
            pa.array(actions, type=schema.field("action").type),
            pa.array(episode_index, type=pa.int64()),
        ],
        schema=schema,
    )
    parquet_path = tmp_path / "actions.parquet"
    pq.write_table(table, parquet_path)

    module.rewrite_action_parquet_file(
        parquet_path,
        gap_threshold=1,
        indices=[6, 8],
        target_values=[0.1, 30.0],
    )

    rewritten_actions = np.array(pq.read_table(parquet_path)["action"].to_pylist(), dtype=np.float32)
    np.testing.assert_array_equal(rewritten_actions[:, 6], np.array([0.1, 0.1, 0.1], dtype=np.float32))
    np.testing.assert_array_equal(rewritten_actions[:, 8], np.array([30.0, 30.0, 30.0], dtype=np.float32))
