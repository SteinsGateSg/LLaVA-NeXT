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
"""
Implement Critic
"""

import os
from collections import defaultdict
from typing import Any, Dict

import torch
import torch.nn.functional as F
from ray.experimental.tqdm_ray import tqdm
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from ...protocol import DataProto
from ...trainer import core_algos
from ...utils import torch_functional as VF
from ...utils.py_functional import append_to_dict
from ...utils.ulysses import gather_outputs_and_unpad, ulysses_pad_and_slice_inputs
from .base import BasePPOCritic
from .config import CriticConfig


try:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
except ImportError:
    pass


__all__ = ["DataParallelPPOCritic"]


class DataParallelPPOCritic(BasePPOCritic):
    def __init__(self, config: CriticConfig, critic_module: nn.Module, critic_optimizer: torch.optim.Optimizer):
        super().__init__(config)
        self.rank = int(os.getenv("RANK", "0"))
        self.critic_module = critic_module
        self.critic_optimizer = critic_optimizer

    def _forward_micro_batch(
        self, micro_batch: Dict[str, torch.Tensor], return_hidden_states: bool = False
    ) -> torch.Tensor:
        input_ids = micro_batch["input_ids"]
        batch_size, seqlen = input_ids.shape
        attention_mask = micro_batch["attention_mask"]
        position_ids = micro_batch["position_ids"]
        responses = micro_batch["responses"]
        response_length = responses.size(-1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch:
            for key in micro_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat(
                    [inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0
                )

        if self.config.padding_free:
            if return_hidden_states:
                raise ValueError("Contrastive loss is not supported with padding_free.")
            input_ids_rmpad, indices, *_ = unpad_input(
                input_ids.unsqueeze(-1), attention_mask
            )  # input_ids_rmpad (total_nnz, ...)
            input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

            # unpad the position_ids to align the rotary
            if position_ids.dim() == 3:
                position_ids_rmpad = (
                    index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                    .transpose(0, 1)
                    .unsqueeze(1)
                )  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
            else:
                position_ids_rmpad = index_first_axis(
                    rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                ).transpose(0, 1)

            # pad and slice the inputs if sp > 1
            if self.config.ulysses_sequence_parallel_size > 1:
                input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad, position_ids_rmpad, sp_size=self.config.ulysses_sequence_parallel_size
                )

            # only pass input_ids and position_ids to enable flash_attn_varlen
            output = self.critic_module(
                input_ids=input_ids_rmpad,
                attention_mask=None,
                position_ids=position_ids_rmpad,
                **multi_modal_inputs,
                use_cache=False,
                output_hidden_states=False,
                return_dict=True,
            )  # prevent model thinks we are generating
            values_rmpad = output.logits
            values_rmpad = values_rmpad.squeeze(0)  # (total_nnz)

            # gather output if sp > 1
            if self.config.ulysses_sequence_parallel_size > 1:
                values_rmpad = gather_outputs_and_unpad(values_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size)

            # pad it back
            values = pad_input(values_rmpad, indices=indices, batch=batch_size, seqlen=seqlen).squeeze(-1)
            values = values[:, -response_length - 1 : -1]
            return values
        else:
            output = self.critic_module(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **multi_modal_inputs,
                use_cache=False,
                output_hidden_states=return_hidden_states,
                return_dict=True,
            )
            values: torch.Tensor = output.logits
            values = values[:, -response_length - 1 : -1].squeeze(-1)  # (bsz, response_length, vocab_size)
            if return_hidden_states:
                hidden_states = output.hidden_states[-1]
                return values, hidden_states
        return values

    def _get_prompt_learner(self) -> nn.Module | None:
        module = self.critic_module
        if isinstance(module, FSDP):
            module = module.module
        return getattr(module, "prompt_learner", None)

    def _masked_mean(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.to(features.dtype)
        mask = mask.unsqueeze(-1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return (features * mask).sum(dim=1) / denom

    def _compute_contrastive_loss(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        response_length: int,
        image_token_id: torch.Tensor | None,
    ) -> torch.Tensor | None:
        prompt_learner = self._get_prompt_learner()
        if prompt_learner is None:
            return None
        if image_token_id is None:
            return None
        image_token_id = image_token_id.view(-1, 1)
        prompt_mask = attention_mask[:, :-response_length].bool()
        prompt_hidden = hidden_states[:, :-response_length]
        image_mask = (input_ids[:, :-response_length] == image_token_id) & prompt_mask
        text_mask = prompt_mask & ~image_mask
        valid_mask = image_mask.any(dim=1) & text_mask.any(dim=1)
        if valid_mask.sum() < 2:
            return None

        text_embeddings = self._masked_mean(prompt_hidden[valid_mask], text_mask[valid_mask])
        image_embeddings = self._masked_mean(prompt_hidden[valid_mask], image_mask[valid_mask])
        prompt_embeddings = prompt_learner(image_embeddings)

        text_embeddings = F.normalize(text_embeddings, dim=-1)
        prompt_embeddings = F.normalize(prompt_embeddings, dim=-1)
        logits = torch.matmul(prompt_embeddings, text_embeddings.transpose(0, 1)) / self.config.contrastive_temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        return F.cross_entropy(logits, labels)

    def _optimizer_step(self) -> torch.Tensor:
        if isinstance(self.critic_module, FSDP):
            grad_norm = self.critic_module.clip_grad_norm_(self.config.max_grad_norm)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.critic_module.parameters(), max_norm=self.config.max_grad_norm
            )

        if not torch.isfinite(grad_norm):
            print("Gradient norm is not finite. Skip update.")
        else:
            self.critic_optimizer.step()

        self.critic_optimizer.zero_grad()
        return grad_norm

    @torch.no_grad()
    def compute_values(self, data: DataProto) -> torch.Tensor:
        self.critic_module.eval()

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        if "multi_modal_inputs" in data.non_tensor_batch.keys():
            non_tensor_select_keys = ["multi_modal_inputs"]
        else:
            non_tensor_select_keys = []

        micro_batches = data.select(select_keys, non_tensor_select_keys).split(
            self.config.micro_batch_size_per_device_for_experience
        )
        values_lst = []
        if self.rank == 0:
            micro_batches = tqdm(micro_batches, desc="Compute values", position=2)

        for micro_batch in micro_batches:
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            values = self._forward_micro_batch(model_inputs)
            values_lst.append(values)

        values = torch.concat(values_lst, dim=0)
        responses = data.batch["responses"]
        attention_mask = data.batch["attention_mask"]
        response_length = responses.size(1)
        values = values * attention_mask[:, -response_length - 1 : -1]
        return values

    def update_critic(self, data: DataProto) -> Dict[str, Any]:
        self.critic_module.train()

        select_keys = [
            "input_ids",
            "responses",
            "attention_mask",
            "position_ids",
            "values",
            "returns",
            "image_token_id",
        ]
        if "multi_modal_inputs" in data.non_tensor_batch.keys():
            non_tensor_select_keys = ["multi_modal_inputs"]
        else:
            non_tensor_select_keys = []

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.select(select_keys, non_tensor_select_keys).split(self.config.global_batch_size_per_device)

        metrics = defaultdict(list)
        for _ in range(self.config.ppo_epochs):
            if self.rank == 0:
                mini_batches = tqdm(mini_batches, desc="Train mini-batches", position=2)

            for mini_batch in mini_batches:
                gradient_accumulation = (
                    self.config.global_batch_size_per_device // self.config.micro_batch_size_per_device_for_update
                )
                micro_batches = mini_batch.split(self.config.micro_batch_size_per_device_for_update)
                if self.rank == 0:
                    micro_batches = tqdm(micro_batches, desc="Update critic", position=3)

                for micro_batch in micro_batches:
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    responses = model_inputs["responses"]
                    attention_mask = model_inputs["attention_mask"]
                    values = model_inputs["values"]
                    returns = model_inputs["returns"]
                    response_length = responses.size(1)
                    action_mask = attention_mask[:, -response_length - 1 : -1]  # shift left for value computation

                    if self.config.enable_contrastive_loss and not self.config.padding_free:
                        vpreds, hidden_states = self._forward_micro_batch(
                            model_inputs, return_hidden_states=True
                        )
                        contrastive_loss = self._compute_contrastive_loss(
                            hidden_states=hidden_states,
                            input_ids=model_inputs["input_ids"],
                            attention_mask=attention_mask,
                            response_length=response_length,
                            image_token_id=model_inputs.get("image_token_id"),
                        )
                    else:
                        vpreds = self._forward_micro_batch(model_inputs)
                        contrastive_loss = None
                    vf_loss, vf_clipfrac = core_algos.compute_value_loss(
                        vpreds=vpreds,
                        returns=returns,
                        values=values,
                        action_mask=action_mask,
                        cliprange_value=self.config.cliprange_value,
                    )
                    if contrastive_loss is not None:
                        loss = (vf_loss + self.config.contrastive_loss_weight * contrastive_loss) / gradient_accumulation
                    else:
                        loss = vf_loss / gradient_accumulation
                    loss.backward()

                    batch_metrics = {
                        "critic/vf_loss": vf_loss.detach().item(),
                        "critic/vf_clipfrac": vf_clipfrac.detach().item(),
                        "critic/vpred_mean": VF.masked_mean(vpreds, action_mask).detach().item(),
                    }
                    if contrastive_loss is not None:
                        batch_metrics["critic/contrastive_loss"] = contrastive_loss.detach().item()
                    append_to_dict(metrics, batch_metrics)

                grad_norm = self._optimizer_step()
                append_to_dict(metrics, {"critic/grad_norm": grad_norm.detach().item()})

        return metrics
