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

from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.scripts import lerobot_train


def test_resolve_processor_pretrained_path_disables_pretrained_processors_for_mixed_action():
    cfg = SmolVLAConfig(use_discrete_base_heads=True)
    cfg.pretrained_path = "lerobot/smolvla_base"

    assert lerobot_train.resolve_processor_pretrained_path(cfg, resume=False) is None


def test_resolve_processor_pretrained_path_disables_pretrained_processors_for_hybrid_action():
    cfg = SmolVLAConfig(use_hybrid_action_heads=True)
    cfg.pretrained_path = "lerobot/smolvla_base"

    assert lerobot_train.resolve_processor_pretrained_path(cfg, resume=False) is None


def test_resolve_processor_pretrained_path_keeps_pretrained_processors_for_default_smolvla():
    cfg = SmolVLAConfig(use_discrete_base_heads=False)
    cfg.pretrained_path = "lerobot/smolvla_base"

    assert lerobot_train.resolve_processor_pretrained_path(cfg, resume=False) == "lerobot/smolvla_base"
