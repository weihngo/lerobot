#!/usr/bin/env python
"""Replace 0.4→0.25 and -0.4→-0.25 in action[6] and action[7] of a LeKiwi dataset.

Usage:
    python examples/lekiwi/merge_speed_levels.py \
        --src-root dataset/lekiwi_banana_and_blue_block \
        --dst-root dataset/lekiwi_banana_and_blue_block_merged
"""

import argparse
import logging
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from lerobot.datasets.dataset_tools import recompute_stats
from lerobot.datasets.lerobot_dataset import LeRobotDataset

REPLACEMENTS = {0.4: 0.25, -0.4: -0.25}
ACTION_INDICES = [6, 7]


def rewrite_parquet(parquet_path: Path) -> None:
    table = pq.read_table(parquet_path)
    schema = table.schema

    action_field_index = schema.get_field_index("action")
    actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)

    for idx in ACTION_INDICES:
        col = actions[:, idx]
        for old_val, new_val in REPLACEMENTS.items():
            mask = np.isclose(col, old_val, atol=1e-6)
            count = int(mask.sum())
            if count > 0:
                logging.info("  action[%d]: replacing %g → %g (%d frames)", idx, old_val, new_val, count)
                col[mask] = new_val

    updated_action_array = pa.array(actions.tolist(), type=schema.field("action").type)
    updated_table = table.set_column(action_field_index, schema.field("action"), updated_action_array)
    pq.write_table(updated_table, parquet_path, compression="snappy")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src-root", type=Path, required=True, help="Source dataset directory")
    parser.add_argument("--dst-root", type=Path, required=True, help="Destination dataset directory")
    parser.add_argument("--repo-id", type=str, default=None, help="Repo id for recomputing stats")
    parser.add_argument("--skip-recompute-stats", action="store_true", help="Skip recomputing stats.json")
    args = parser.parse_args()

    if not args.src_root.exists():
        raise FileNotFoundError(f"Source dataset does not exist: {args.src_root}")
    if args.dst_root.exists():
        raise FileExistsError(f"Destination already exists: {args.dst_root}")

    logging.info("Copying dataset from %s to %s", args.src_root, args.dst_root)
    shutil.copytree(args.src_root, args.dst_root)

    parquet_files = sorted((args.dst_root / "data").glob("*/*.parquet"))
    if not parquet_files:
        raise ValueError(f"No parquet files found under {args.dst_root / 'data'}")

    for parquet_path in parquet_files:
        logging.info("Rewriting %s", parquet_path)
        rewrite_parquet(parquet_path)

    if args.skip_recompute_stats:
        logging.info("Skipping stats recomputation")
        return

    dataset = LeRobotDataset(repo_id=args.repo_id or args.dst_root.name, root=args.dst_root)
    recompute_stats(dataset, skip_image_video=True)
    logging.info("Done. Stats recomputed.")


if __name__ == "__main__":
    main()
