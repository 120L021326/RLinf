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

import os
import sys
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig

from rlinf.config import torch_dtype_from_precision
from rlinf.data.io_struct import RolloutRequest, RolloutResult
from rlinf.scheduler import Channel, Worker
from rlinf.utils.placement import ModelParallelComponentPlacement
from rlinf.workers.rollout.utils import RankMapper

ALPAMAYO_SRC = Path(__file__).resolve().parents[5] / "alpamayo" / "src"
if str(ALPAMAYO_SRC) not in sys.path:
    sys.path.insert(0, str(ALPAMAYO_SRC))

from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1


class AlpamayoWorker(Worker):
    """Minimal Alpamayo rollout worker for CoT-only reasoning RL."""

    def __init__(
        self,
        config: DictConfig,
        placement: ModelParallelComponentPlacement,
        config_rollout: DictConfig = None,
    ):
        Worker.__init__(self)
        self._cfg = config
        self._placement = placement
        if config_rollout is None:
            config_rollout = self._cfg.rollout
        self._cfg_rollout = config_rollout
        self._return_logprobs = bool(self._cfg_rollout.return_logprobs)
        self._sampling_params = self.get_sampling_param_from_config(
            self._cfg.algorithm.sampling_params
        )

    def get_sampling_param_from_config(
        self, cfg_sampling_params: DictConfig
    ) -> dict[str, Any]:
        return {
            "temperature": cfg_sampling_params.temperature,
            "top_k": cfg_sampling_params.top_k,
            "top_p": cfg_sampling_params.top_p,
            "repetition_penalty": cfg_sampling_params.repetition_penalty,
            "max_new_tokens": cfg_sampling_params.max_new_tokens,
            "min_new_tokens": cfg_sampling_params.get("min_new_tokens", None),
            "stop_token_ids": cfg_sampling_params.get("stop_token_ids", None),
        }

    def init_worker(self):
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        Worker.torch_platform.set_device(local_rank)
        self.device = Worker.torch_platform.current_device()

        self.model = AlpamayoR1.from_pretrained(
            self._cfg_rollout.model.model_path,
            torch_dtype=torch_dtype_from_precision(self._cfg_rollout.model.precision),
        ).to(self.device)
        self.model.eval()

        rollout_tp_size = self._placement.rollout_tp_size
        rollout_rank = (self._rank // rollout_tp_size, self._rank % rollout_tp_size)
        rollout_to_actor_rank = RankMapper.get_rollout_rank_to_actor_rank_map(
            self._placement
        )
        self.actor_weight_src_rank = rollout_to_actor_rank[rollout_rank]
        self.log_info(
            f"Rollout rank {self._rank} will receive weights from actor rank {self.actor_weight_src_rank}"
        )

    def _move_multi_modal_inputs_to_device(
        self, multi_modal_inputs: dict[str, Any] | None
    ) -> dict[str, Any]:
        if multi_modal_inputs is None:
            return {}
        result = {}
        for key, value in multi_modal_inputs.items():
            if isinstance(value, torch.Tensor):
                result[key] = value.to(self.device)
            else:
                result[key] = value
        return result

    def _generate_group(self, request_group) -> RolloutResult:
        input_ids = torch.as_tensor(
            request_group.input_ids, dtype=torch.long, device=self.device
        ).unsqueeze(0)
        multi_modal_inputs = self._move_multi_modal_inputs_to_device(
            request_group.multi_modal_inputs
        )

        with torch.no_grad():
            generate_result = self.model.generate_cot_only_from_fused_prompt(
                input_ids=input_ids,
                multi_modal_inputs=multi_modal_inputs,
                top_p=self._sampling_params["top_p"],
                top_k=self._sampling_params["top_k"],
                temperature=self._sampling_params["temperature"],
                repetition_penalty=self._sampling_params["repetition_penalty"],
                num_return_sequences=request_group.group_size,
                max_generation_length=self._sampling_params["max_new_tokens"],
                min_generation_length=self._sampling_params["min_new_tokens"],
                stop_token_ids=self._sampling_params["stop_token_ids"],
                return_logprobs=self._return_logprobs,
            )

        rollout_result = RolloutResult(
            num_sequence=request_group.group_size,
            group_size=request_group.group_size,
            prompt_lengths=[len(request_group.input_ids)] * request_group.group_size,
            prompt_ids=[request_group.input_ids] * request_group.group_size,
            response_lengths=[
                len(token_ids) for token_ids in generate_result["response_ids"]
            ],
            response_ids=generate_result["response_ids"],
            response_texts=generate_result["response_texts"],
            answers=[request_group.answer] * request_group.group_size,
            image_data=[request_group.image_data] * request_group.group_size,
            multi_modal_inputs=[
                request_group.multi_modal_inputs
            ]
            * request_group.group_size,
            is_end=generate_result["is_end"],
        )
        if self._return_logprobs:
            rollout_result.rollout_logprobs = generate_result["rollout_logprobs"]
        return rollout_result

    async def sync_model_from_actor(self):
        state_dict = await self.recv(
            self._cfg.actor.group_name,
            src_rank=self.actor_weight_src_rank,
            async_op=True,
        ).async_wait()
        bucket_length = state_dict.pop("bucket_length", 1)
        for _ in range(bucket_length - 1):
            next_bucket = await self.recv(
                self._cfg.actor.group_name,
                src_rank=self.actor_weight_src_rank,
                async_op=True,
            ).async_wait()
            state_dict.update(next_bucket)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()

    async def rollout(self, input_channel: Channel, output_channel: Channel):
        request: RolloutRequest = input_channel.get()
        groups = request.to_seq_group_infos()
        with self.device_lock, self.worker_timer():
            for group in groups:
                rollout_result = self._generate_group(group)
                await output_channel.put(
                    item=rollout_result, async_op=True
                ).async_wait()
