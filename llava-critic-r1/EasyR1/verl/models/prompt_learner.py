# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class PromptLearnerWrapper(nn.Module):
    def __init__(self, model: nn.Module, hidden_size: int, prompt_hidden_size: Optional[int] = None) -> None:
        super().__init__()
        self.model = model
        inner_size = prompt_hidden_size or hidden_size
        self.prompt_learner = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, inner_size, bias=False),
            nn.GELU(),
            nn.Linear(inner_size, hidden_size, bias=False),
        )

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def compute_prompt_embeddings(self, image_embeddings: torch.Tensor) -> torch.Tensor:
        return self.prompt_learner(image_embeddings)
