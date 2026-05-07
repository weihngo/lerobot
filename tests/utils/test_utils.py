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

import logging

from lerobot.utils.utils import init_logging


def test_init_logging_creates_parent_directory_for_log_file(tmp_path):
    log_file = tmp_path / "nested" / "train.log"

    init_logging(log_file=log_file)
    logging.info("hello from test")

    for handler in logging.getLogger().handlers:
        handler.flush()

    assert log_file.exists()
    assert "hello from test" in log_file.read_text()
