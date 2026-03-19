#!/usr/bin/env python3
# Copyright 2025-2026 Dimensional Inc.
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

"""Blueprint using any OpenAI-compatible remote API (e.g. vLLM, SGLang serving Qwen/etc.).

Required environment variables:
    OPENAI_BASE_URL   - Base URL of the OpenAI-compatible server, e.g.
                        http://192.168.1.100:8000/v1
    OPENAI_API_KEY    - API key (can be any non-empty string for local servers)
    DIMOS_MODEL       - (optional) model name as served by the remote server,
                        defaults to "qwen2.5-235b-instruct"

Example:
    export OPENAI_BASE_URL=http://192.168.1.100:8000/v1
    export OPENAI_API_KEY=any-key
    export DIMOS_MODEL=qwen2.5-235b-instruct
    dimos --replay run unitree-go2-agentic-openai-compat
"""

import os

from dimos.agents.agent import agent
from dimos.core.blueprints import autoconnect
from dimos.robot.unitree.go2.blueprints.agentic._common_agentic import _common_agentic
from dimos.robot.unitree.go2.blueprints.smart.unitree_go2_spatial import unitree_go2_spatial

_model_name = os.environ.get("DIMOS_MODEL", "qwen2.5-235b-instruct")

unitree_go2_agentic_openai_compat = autoconnect(
    unitree_go2_spatial,
    agent(model=f"openai:{_model_name}"),
    _common_agentic,
)

__all__ = ["unitree_go2_agentic_openai_compat"]
