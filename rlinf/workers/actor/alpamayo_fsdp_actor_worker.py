# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
from pathlib import Path

import torch
from torch import nn

from .fsdp_actor_worker import FSDPActor

ALPAMAYO_SRC = Path(__file__).resolve().parents[4] / "alpamayo" / "src"
if str(ALPAMAYO_SRC) not in sys.path:
    sys.path.insert(0, str(ALPAMAYO_SRC))

from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1


class AlpamayoFSDPActor(FSDPActor):
    """FSDP actor for CoT-only Alpamayo RL training."""

    def model_provider_func(self) -> nn.Module:
        freeze_trajectory_modules = self.cfg.actor.model.get(
            "freeze_trajectory_modules_for_rl", True
        )
        if freeze_trajectory_modules:
            # FSDP flatten groups require uniform requires_grad when use_orig_params=False.
            self._cfg.fsdp_config.use_orig_params = True
            self.cfg.actor.fsdp_config.use_orig_params = True

        model = AlpamayoR1.from_pretrained(
            self.cfg.actor.model.model_path,
            torch_dtype=self.torch_dtype,
        )

        if freeze_trajectory_modules:
            model.freeze_trajectory_modules_for_rl()

        if self.cfg.runner.get("ckpt_path", None):
            model_dict = torch.load(self.cfg.runner.ckpt_path)
            model.load_state_dict(model_dict)

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        return model
