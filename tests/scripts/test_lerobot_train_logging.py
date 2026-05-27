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

import lerobot.scripts.lerobot_train as lerobot_train


def test_format_train_log_message_appends_act_sub_losses():
    message = lerobot_train.format_train_log_message(
        "step:24K loss:0.256",
        {
            "arm_l1_loss": 0.18123,
            "base_bce_loss": 0.07456,
            "kld_loss": 0.00321,
        },
    )

    assert message == (
        "step:24K loss:0.256 "
        "arm_l1_loss:0.181 base_bce_loss:0.075 kld_loss:0.003"
    )


def test_format_train_log_message_ignores_non_act_fields():
    message = lerobot_train.format_train_log_message(
        "step:24K loss:0.256",
        {
            "l1_loss": 0.2,
            "foo": 1.0,
            "base_bce_loss": 0.07456,
            "note": "ignored",
        },
    )

    assert message == "step:24K loss:0.256 l1_loss:0.200 base_bce_loss:0.075"
