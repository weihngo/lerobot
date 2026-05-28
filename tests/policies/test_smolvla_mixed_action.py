import sys
from pathlib import Path
import importlib

# Ensure the local worktree 'src' is prioritized on sys.path so tests import
# the current worktree's code instead of any external copies.
_repo_root = Path(__file__).resolve().parents[2]
_src = _repo_root / "src"
_src_str = str(_src)
if _src_str not in sys.path:
    sys.path.insert(0, _src_str)

# If another copy of 'lerobot' was already imported (e.g., from a different
# checkout installed in the environment), remove those modules so the import
# below loads from the worktree's src. Invalidate caches to ensure fresh import.
for m in list(sys.modules):
    if m == "lerobot" or m.startswith("lerobot."):
        del sys.modules[m]
importlib.invalidate_caches()

from lerobot.policies.smolvla.configuration_smolvla import (
    SmolVLAActionHeadConfig,
    SmolVLAConfig,
)
from lerobot.configs.policies import PreTrainedConfig


def test_smolvla_mixed_action_defaults_define_three_base_heads():
    config = SmolVLAConfig(
        use_discrete_base_heads=True,
        arm_passthrough_dims=[0, 1, 2, 3, 4, 5],
        export_action_dim=9,
    )

    assert config.base_action_dims == [6, 7, 8]
    assert [head.name for head in config.action_heads] == ["base_x", "base_y", "base_theta"]
    assert config.action_heads[0].values == [-0.25, 0.0, 0.25]


def test_smolvla_mixed_action_rejects_overlapping_indices():
    try:
        SmolVLAConfig(
            use_discrete_base_heads=True,
            arm_passthrough_dims=[0, 1, 2, 3, 4, 6],
            action_heads=[
                SmolVLAActionHeadConfig(
                    name="base_x",
                    head_type="categorical",
                    index=6,
                    values=[-0.25, 0.0, 0.25],
                )
            ],
        )
    except ValueError as exc:
        assert "overlap" in str(exc).lower()
    else:
        raise AssertionError("Expected overlap validation to fail")


def test_smolvla_mixed_action_rejects_duplicate_head_indices():
    try:
        SmolVLAConfig(
            use_discrete_base_heads=True,
            action_heads=[
                SmolVLAActionHeadConfig("base_x", "categorical", 6, [-0.25, 0.0, 0.25]),
                SmolVLAActionHeadConfig("base_x_dup", "categorical", 6, [-0.25, 0.0, 0.25]),
            ],
        )
    except ValueError as exc:
        assert "duplicate" in str(exc).lower()
    else:
        raise AssertionError("Expected duplicate head index validation to fail")


def test_smolvla_mixed_action_rejects_base_action_dims_mismatch():
    try:
        SmolVLAConfig(
            use_discrete_base_heads=True,
            base_action_dims=[6, 7, 9],
        )
    except ValueError as exc:
        assert "base_action_dims" in str(exc)
    else:
        raise AssertionError("Expected base_action_dims mismatch validation to fail")


def test_smolvla_mixed_action_rejects_base_action_dims_duplicate_values():
    try:
        SmolVLAConfig(
            use_discrete_base_heads=True,
            base_action_dims=[6, 6, 7, 8],
        )
    except ValueError as exc:
        assert "duplicate" in str(exc).lower()
    else:
        raise AssertionError("Expected duplicate base_action_dims validation to fail")


def test_smolvla_mixed_action_rejects_sum_mismatch():
    try:
        SmolVLAConfig(
            use_discrete_base_heads=True,
            arm_passthrough_dims=[0, 1],
        )
    except ValueError as exc:
        assert "must equal export_action_dim" in str(exc)
    else:
        raise AssertionError("Expected sum mismatch validation to fail")


def test_smolvla_mixed_action_rejects_head_index_out_of_range():
    try:
        SmolVLAConfig(
            use_discrete_base_heads=True,
            action_heads=[SmolVLAActionHeadConfig("bad", "categorical", 9, [0.0])],
        )
    except ValueError as exc:
        assert "out of range" in str(exc).lower()
    else:
        raise AssertionError("Expected head index out of range validation to fail")


def test_smolvla_mixed_action_rejects_arm_index_out_of_range():
    try:
        SmolVLAConfig(
            use_discrete_base_heads=True,
            arm_passthrough_dims=[0, 1, 2, 3, 4, 9],
        )
    except ValueError as exc:
        assert "out of range" in str(exc).lower()
    else:
        raise AssertionError("Expected arm passthrough index out of range validation to fail")


def test_smolvla_mixed_action_save_load_roundtrip(tmp_path):
    cfg = SmolVLAConfig(
        use_discrete_base_heads=True,
        arm_passthrough_dims=[0, 1, 2, 3, 4, 5],
        export_action_dim=9,
    )

    # save + load round-trip should succeed and preserve action heads
    cfg._save_pretrained(tmp_path)
    loaded = PreTrainedConfig.from_pretrained(tmp_path)
    assert isinstance(loaded, SmolVLAConfig)
    assert [h.name for h in loaded.action_heads] == [h.name for h in cfg.action_heads]


def test_smolvla_mixed_action_rejects_arm_passthrough_duplicate_values():
    try:
        SmolVLAConfig(
            use_discrete_base_heads=True,
            arm_passthrough_dims=[0, 1, 2, 2, 4, 5],
            export_action_dim=9,
        )
    except ValueError as exc:
        assert "duplicate" in str(exc).lower()
    else:
        raise AssertionError("Expected duplicate arm_passthrough_dims validation to fail")
