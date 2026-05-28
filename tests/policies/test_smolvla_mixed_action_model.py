import importlib
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_repo_root = Path(__file__).resolve().parents[2]
_src = _repo_root / "src"
_src_str = str(_src)
if _src_str not in sys.path:
    sys.path.insert(0, _src_str)

for m in list(sys.modules):
    if m == "lerobot" or m.startswith("lerobot."):
        del sys.modules[m]
importlib.invalidate_caches()

from lerobot.policies.smolvla.configuration_smolvla import SmolVLAActionHeadConfig, SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.processor_smolvla import SMOLVLA_ARM_STATE_KEY
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE


class FakeVLAFlowMatching:
    def __init__(self, config, rtc_processor=None):
        self.config = config

    def forward(self, images, img_masks, lang_tokens, lang_masks, state, actions, noise=None, time=None):
        batch_size = state.shape[0]
        chunk_size = actions.shape[1]
        if self.config.use_discrete_base_heads:
            base_x = torch.tensor([[[0.0, 0.0, 4.0]]], dtype=torch.float32).expand(batch_size, chunk_size, -1)
            base_y = torch.tensor([[[4.0, 0.0, 0.0]]], dtype=torch.float32).expand(batch_size, chunk_size, -1)
            base_theta = torch.tensor([[[0.0, 5.0, 0.0]]], dtype=torch.float32).expand(batch_size, chunk_size, -1)
            return {
                "base_x_logits": base_x.clone(),
                "base_y_logits": base_y.clone(),
                "base_theta_logits": base_theta.clone(),
            }
        return torch.ones(batch_size, chunk_size, self.config.max_action_dim, dtype=torch.float32)

    def sample_actions(self, images, img_masks, lang_tokens, lang_masks, state, noise=None, **kwargs):
        batch_size = state.shape[0]
        chunk_size = self.config.chunk_size
        if not self.config.use_discrete_base_heads:
            return torch.zeros(batch_size, chunk_size, self.config.max_action_dim, dtype=torch.float32)
        return {
            "base_x_logits": torch.tensor([[[0.0, 0.0, 4.0]]], dtype=torch.float32).expand(batch_size, chunk_size, -1),
            "base_y_logits": torch.tensor([[[4.0, 0.0, 0.0]]], dtype=torch.float32).expand(batch_size, chunk_size, -1),
            "base_theta_logits": torch.tensor([[[0.0, 5.0, 0.0]]], dtype=torch.float32).expand(batch_size, chunk_size, -1),
        }


def _make_policy(monkeypatch, *, use_discrete: bool) -> SmolVLAPolicy:
    import lerobot.policies.smolvla.modeling_smolvla as modeling

    monkeypatch.setattr(modeling, "VLAFlowMatching", FakeVLAFlowMatching)
    cfg = SmolVLAConfig(
        use_discrete_base_heads=use_discrete,
        chunk_size=2,
        n_action_steps=2,
        max_action_dim=9,
        max_state_dim=9,
        action_heads=[
            SmolVLAActionHeadConfig("base_x", "categorical", 6, [-0.25, 0.0, 0.25], loss_weight=1.0),
            SmolVLAActionHeadConfig("base_y", "categorical", 7, [-0.25, 0.0, 0.25], loss_weight=2.0),
            SmolVLAActionHeadConfig("base_theta", "categorical", 8, [-0.25, 0.0, 0.25], loss_weight=0.5),
        ],
    )
    cfg.input_features = {"observation.state": PolicyFeature(type=FeatureType.STATE, shape=(9,))}
    cfg.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(9,))}
    policy = SmolVLAPolicy(cfg)
    policy.prepare_images = lambda batch: ([torch.zeros(batch[OBS_STATE].shape[0], 3, 2, 2)], [torch.ones(batch[OBS_STATE].shape[0], dtype=torch.bool)])  # type: ignore[method-assign]
    policy.prepare_state = lambda batch: batch[OBS_STATE]  # type: ignore[method-assign]
    policy.prepare_action = lambda batch: torch.zeros(batch[OBS_STATE].shape[0], cfg.chunk_size, cfg.max_action_dim)  # type: ignore[method-assign]
    return policy


def test_mixed_action_forward_computes_weighted_cross_entropy(monkeypatch):
    policy = _make_policy(monkeypatch, use_discrete=True)
    batch = {
        OBS_STATE: torch.zeros(1, 9),
        OBS_LANGUAGE_TOKENS: torch.ones(1, 1, dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(1, 1, dtype=torch.bool),
        ACTION: torch.zeros(1, 2, 9),
        "base_x_target": torch.tensor([[2, 2]], dtype=torch.long),
        "base_y_target": torch.tensor([[0, 0]], dtype=torch.long),
        "base_theta_target": torch.tensor([[1, 1]], dtype=torch.long),
    }

    loss, loss_dict = policy.forward(batch)

    expected = (
        F.cross_entropy(torch.tensor([[0.0, 0.0, 4.0]]), torch.tensor([2])) * 1.0
        + F.cross_entropy(torch.tensor([[4.0, 0.0, 0.0]]), torch.tensor([0])) * 2.0
        + F.cross_entropy(torch.tensor([[0.0, 5.0, 0.0]]), torch.tensor([1])) * 0.5
    ) / 3.5
    assert torch.isclose(loss, expected)
    assert set(loss_dict) >= {"base_x_ce", "base_y_ce", "base_theta_ce", "loss"}


def test_mixed_action_get_action_chunk_decodes_logits_to_full_action(monkeypatch):
    policy = _make_policy(monkeypatch, use_discrete=True)
    raw_state = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0, 0.0, 0.0]])
    batch = {
        OBS_STATE: torch.zeros(1, 9),
        OBS_LANGUAGE_TOKENS: torch.ones(1, 1, dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(1, 1, dtype=torch.bool),
        SMOLVLA_ARM_STATE_KEY: raw_state,
    }

    actions = policy._get_action_chunk(batch)

    expected = torch.tensor(
        [
            [
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.25, -0.25, 0.0],
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.25, -0.25, 0.0],
            ]
        ]
    )
    torch.testing.assert_close(actions, expected)


def test_default_path_untouched_when_use_discrete_false(monkeypatch):
    policy = _make_policy(monkeypatch, use_discrete=False)
    batch = {
        OBS_STATE: torch.zeros(1, 9),
        OBS_LANGUAGE_TOKENS: torch.ones(1, 1, dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(1, 1, dtype=torch.bool),
        ACTION: torch.zeros(1, 2, 9),
    }

    loss, _ = policy.forward(batch)
    actions = policy._get_action_chunk(batch)

    assert torch.isclose(loss, torch.tensor(1.0))
    assert actions.shape == (1, 2, 9)
