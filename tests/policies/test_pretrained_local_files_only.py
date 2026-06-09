from dataclasses import dataclass

import torch
from torch import Tensor, nn

from lerobot.configs.policies import PreTrainedConfig
from lerobot.optim.optimizers import OptimizerConfig
from lerobot.policies.pretrained import PreTrainedPolicy


@dataclass
class TinyPolicyConfig(PreTrainedConfig):
    def get_optimizer_preset(self) -> OptimizerConfig:
        raise NotImplementedError

    def get_scheduler_preset(self):
        return None

    def validate_features(self) -> None:
        pass

    @property
    def observation_delta_indices(self) -> list | None:
        return None

    @property
    def action_delta_indices(self) -> list | None:
        return None

    @property
    def reward_delta_indices(self) -> list | None:
        return None


class TinyPolicy(PreTrainedPolicy):
    config_class = TinyPolicyConfig
    name = "tiny"

    def __init__(self, config: TinyPolicyConfig, *, local_files_only: bool = False):
        super().__init__(config)
        self.local_files_only = local_files_only
        self.weight = nn.Parameter(torch.zeros(1))

    def get_optim_params(self) -> dict:
        return {"params": self.parameters()}

    def reset(self):
        pass

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
        return self.weight.sum(), None

    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        return self.weight

    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        return self.weight


def test_from_pretrained_forwards_local_files_only_to_policy_init(monkeypatch, tmp_path):
    config = TinyPolicyConfig(device="cpu")
    model_dir = tmp_path / "policy"
    model_dir.mkdir()

    def fake_load_as_safetensor(model, model_file, map_location, strict):
        return model

    monkeypatch.setattr(TinyPolicy, "_load_as_safetensor", staticmethod(fake_load_as_safetensor))

    policy = TinyPolicy.from_pretrained(
        model_dir,
        config=config,
        local_files_only=True,
    )

    assert policy.local_files_only is True
