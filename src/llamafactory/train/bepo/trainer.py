# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's TRL library.
# https://github.com/huggingface/trl/blob/v0.8.0/trl/trainer/dpo_trainer.py
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

import warnings
from collections import defaultdict
from contextlib import nullcontext
from types import MethodType
from typing import TYPE_CHECKING, Dict, Literal, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from transformers import Trainer
from trl import DPOTrainer
from trl.trainer import disable_dropout_in_model
from typing_extensions import override

from ...extras.constants import IGNORE_INDEX
from ...extras.logging import get_logger
from ..callbacks import PissaConvertCallback, SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler, get_batch_logps
from ..ppl.ppl_loss import compute_ensemble_loss
from ..ppl.trainer import initialize_prior_head, get_last_hidden_state_before_label
import os

logger = get_logger(__name__)


if TYPE_CHECKING:
    from transformers import PreTrainedModel, ProcessorMixin

    from ...hparams import FinetuningArguments


class CustomBEPOTrainer(DPOTrainer):
    def __init__(
        self,
        model: Union["PreTrainedModel", torch.nn.Module],
        ref_model: Optional[Union["PreTrainedModel", torch.nn.Module]],
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        disable_dropout: bool = True,
        **kwargs,
    ):
        if disable_dropout:
            disable_dropout_in_model(model)
            if ref_model is not None:
                disable_dropout_in_model(ref_model)

        self.finetuning_args = finetuning_args
        self.f_divergence_type = "reverse_kl"
        self.reference_free = False
        self.use_dpo_data_collator = True  # hack to avoid warning
        self.generate_during_eval = False  # disable at evaluation
        self.label_pad_token_id = IGNORE_INDEX
        self.padding_value = 0
        self.is_encoder_decoder = model.config.is_encoder_decoder
        self.precompute_ref_log_probs = False
        self._precomputed_train_ref_log_probs = False
        self._precomputed_eval_ref_log_probs = False
        self._peft_has_been_casted_to_bf16 = False

        self.ref_model = ref_model
        self._stored_metrics = defaultdict(lambda: defaultdict(list))

        # dpo hyperparams
        self.beta = finetuning_args.pref_beta
        self.loss_type = finetuning_args.pref_loss
        self.ftx_gamma = finetuning_args.pref_ftx
        self.label_smoothing = finetuning_args.dpo_label_smoothing
        self.simpo_gamma = finetuning_args.simpo_gamma

        Trainer.__init__(self, model=model, **kwargs)
        if not hasattr(self, "accelerator"):
            raise AttributeError("Please update `transformers`.")

        warnings.simplefilter("ignore")  # remove gc warnings on ref model

        if ref_model is not None:
            if self.is_deepspeed_enabled:
                if not (
                    getattr(ref_model, "is_loaded_in_8bit", False) or getattr(ref_model, "is_loaded_in_4bit", False)
                ):  # quantized models are already set on the correct device
                    self.ref_model = self._prepare_deepspeed(self.ref_model)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
                self.ref_model.eval()

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.pissa_convert:
            self.callback_handler.add_callback(PissaConvertCallback)

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)
        
        # Initialize prior head
        self.prior_head = initialize_prior_head(finetuning_args, hidden_size=self.model.config.hidden_size)
        if self.prior_head is not None:
            # Prepare prior_head with accelerator for proper distributed training and mixed precision
            self.prior_head = self.accelerator.prepare_model(self.prior_head, evaluation_mode=False)

        # Initialize reference prior head
        self.ref_prior_head = initialize_prior_head(finetuning_args, hidden_size=self.ref_model.config.hidden_size)
        if self.ref_prior_head is not None:
            self.ref_prior_head = self.accelerator.prepare_model(self.ref_prior_head, evaluation_mode=True)
            self.ref_prior_head.eval()
        
        # Initialize prior loss function if using logistic loss
        if finetuning_args.ppl_prior_loss_type in ['logistic', 'logistic+llk']:
            self.prior_loss_fn = torch.nn.BCEWithLogitsLoss()
        
        print(f"[BEPO Trainer - Prior Loss] Prior loss type: {finetuning_args.ppl_prior_loss_type}")
        print(f"[BEPO Trainer - Prior Loss] Use prior head loss: {finetuning_args.use_prior_head_loss}")
        print(f"[BEPO Trainer - Prior Loss] Prior head loss factor: {finetuning_args.ppl_prior_loss_factor}")

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        
        # Call parent to create optimizer if create_custom_optimizer returned None
        super().create_optimizer()
        
        # Add prior_head parameters to the optimizer if it exists and is trainable
        if self.prior_head is not None and self.optimizer is not None:
            prior_head_params = list(self.prior_head.parameters())
            if prior_head_params and prior_head_params[0].requires_grad:
                # Check if prior_head params are already in optimizer
                optimizer_params = set()
                for group in self.optimizer.param_groups:
                    optimizer_params.update(id(p) for p in group['params'])
                
                prior_head_param_ids = set(id(p) for p in prior_head_params)
                
                if not prior_head_param_ids.issubset(optimizer_params):
                    # Add prior_head parameters as a new param group
                    logger.info(f"Adding {len(prior_head_params)} prior_head parameters to optimizer. LR={self.finetuning_args.prior_head_lr}")
                    self.optimizer.add_param_group({
                        'params': prior_head_params,
                        'lr': self.finetuning_args.prior_head_lr,
                        'weight_decay': self.args.weight_decay,
                    })
                else:
                    logger.info("Prior_head parameters already in optimizer")
        
        # Report total trainable parameters in optimizer
        logger.info("=" * 80)
        logger.info("OPTIMIZER PARAMETER SUMMARY")
        logger.info("=" * 80)
        
        total_params = 0
        total_trainable = 0
        
        for group_idx, param_group in enumerate(self.optimizer.param_groups):
            group_params = param_group['params']
            group_total = sum(p.numel() for p in group_params)
            group_trainable = sum(p.numel() for p in group_params if p.requires_grad)
            
            logger.info(f"Param Group {group_idx}:")
            logger.info(f"  Total parameters: {group_total:,}")
            logger.info(f"  Trainable parameters: {group_trainable:,}")
            logger.info(f"  Learning rate: {param_group.get('lr', 'N/A')}")
            logger.info(f"  Weight decay: {param_group.get('weight_decay', 'N/A')}")
            
            total_params += group_total
            total_trainable += group_trainable
        
        logger.info("-" * 80)
        logger.info(f"TOTAL PARAMETERS IN OPTIMIZER: {total_params:,}")
        logger.info(f"TOTAL TRAINABLE PARAMETERS: {total_trainable:,}")
        
        # Verify prior_head parameters are included
        if self.prior_head is not None:
            prior_head_params_count = sum(p.numel() for p in self.prior_head.parameters())
            prior_head_param_ids = {id(p) for p in self.prior_head.parameters()}
            optimizer_param_ids = {id(p) for group in self.optimizer.param_groups for p in group['params']}
            prior_head_in_optimizer = prior_head_param_ids.issubset(optimizer_param_ids)
            
            logger.info(f"Prior head total parameters: {prior_head_params_count:,}")
            logger.info(f"Prior head parameters in optimizer: {prior_head_in_optimizer}")
            
            if not prior_head_in_optimizer:
                logger.warning("⚠️  WARNING: Prior head parameters are NOT in optimizer!")
        
        logger.info("=" * 80)
        
        return self.optimizer

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    def odds_ratio_loss(self, chosen_logps: "torch.Tensor", rejected_logps: "torch.Tensor") -> "torch.Tensor":
        r"""
        Computes ORPO's odds ratio (OR) loss for batched log probabilities of the policy model.
        """
        log_odds = (chosen_logps - rejected_logps) - (
            torch.log1p(-torch.exp(chosen_logps)) - torch.log1p(-torch.exp(rejected_logps))
        )
        sft_loss = -chosen_logps
        odds_ratio_loss = -F.logsigmoid(log_odds)
        orpo_loss = sft_loss + self.beta * odds_ratio_loss
        return orpo_loss

    def simpo_loss(self, chosen_logps: "torch.Tensor", rejected_logps: "torch.Tensor") -> "torch.Tensor":
        r"""
        Computes SimPO loss for batched log probabilities of the policy model.
        """
        pi_logratios = chosen_logps - rejected_logps
        gamma_logratios = self.simpo_gamma / self.beta
        logits = pi_logratios - gamma_logratios
        simpo_loss = -F.logsigmoid(self.beta * logits)
        return simpo_loss
    
    def compute_prior_loss(
        self,
        policy_prior_logits: "torch.Tensor",
        prior_logprob: Optional["torch.Tensor"] = None,
    ) -> "torch.Tensor":
        r"""
        Computes auxiliary prior loss to train the prior head.
        Assumes idx=0 in the batch holds the ground-truth passage.
        """
        if not self.finetuning_args.use_prior_head_loss:
            return torch.tensor(0.0, device=self.accelerator.device)
        
        # Determine loss factor
        prior_lambda = self.finetuning_args.ppl_prior_loss_factor
        if prior_lambda < 0:
            # Use a default value or compute dynamically if needed
            prior_lambda = 1.0
        
        prior_loss = torch.tensor(0.0, device=self.accelerator.device)
        
        if self.finetuning_args.ppl_prior_loss_type == 'softmax':
            # Use log probability from softmax (requires prior_logprob)
            if prior_logprob is not None:
                prior_loss = -prior_logprob[0] * prior_lambda
        elif self.finetuning_args.ppl_prior_loss_type == 'logistic':
            # Binary classification: idx=0 is positive (GT passage), rest are negative
            prior_labels = torch.zeros_like(policy_prior_logits)
            prior_labels[0] = 1.0  # Ground-truth passage is at index 0
            prior_loss = self.prior_loss_fn(policy_prior_logits, prior_labels) * prior_lambda
            # print(f"prior_loss: {prior_loss.detach().cpu()}")
            # print(f"policy_prior_logits: {policy_prior_logits.detach().cpu()}")
            # print(f"prior_labels: {prior_labels.detach().cpu()}")
            # breakpoint()
        else:
            # Default: no prior loss
            pass
        
        return prior_loss

    def compute_bepo_preference_loss(
        self,
        policy_chosen_logits: "torch.Tensor",
        policy_rejected_logits: "torch.Tensor",
        policy_prior_logits: Optional["torch.Tensor"],
        reference_chosen_logits: Optional["torch.Tensor"],
        reference_rejected_logits: Optional["torch.Tensor"],
        reference_prior_logits: Optional["torch.Tensor"],
        labels: "torch.Tensor",
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        r"""
        Computes loss for Baysian Ensemble preference learning. That is, replace the likelihood with the posterior.
        # """
        # if not self.finetuning_args.use_ref_model: #NOTE not implemented
        #     if self.loss_type == "orpo":
        #         losses = self.odds_ratio_loss(policy_chosen_logps, policy_rejected_logps)
        #     elif self.loss_type == "simpo":
        #         losses = self.simpo_loss(policy_chosen_logps, policy_rejected_logps)
        #     else:
        #         raise NotImplementedError("Unknown loss type: {}.".format(self.loss_type))

        #     chosen_rewards = self.beta * policy_chosen_logps.to(self.accelerator.device).detach()
        #     rejected_rewards = self.beta * policy_rejected_logps.to(self.accelerator.device).detach()
        # else:
        K = policy_chosen_logits.shape[0] 

        policy_chosen_posterior, *_ = compute_ensemble_loss(policy_chosen_logits, labels[:K], policy_prior_logits[:K])
        ref_chosen_posterior, *_ = compute_ensemble_loss(reference_chosen_logits, labels[:K], reference_prior_logits[:K])

        policy_rejected_posterior, *_ = compute_ensemble_loss(policy_rejected_logits, labels[K:], policy_prior_logits[K:])
        ref_rejected_posterior, *_ = compute_ensemble_loss(reference_rejected_logits, labels[K:], reference_prior_logits[K:])

        chosen_rewards = -self.beta * (policy_chosen_posterior - ref_chosen_posterior) #NOTE minus sign due to compute_ensemble_loss returning negative log-likelihood
        rejected_rewards = -self.beta * (policy_rejected_posterior - ref_rejected_posterior)
        losses = -F.logsigmoid(chosen_rewards - rejected_rewards)

        return losses, chosen_rewards, rejected_rewards, -policy_chosen_posterior, -policy_rejected_posterior

    @override
    def concatenated_forward(
        self, model: "PreTrainedModel", batch: Dict[str, "torch.Tensor"],
        hidden_state_offset: int = 0
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        r"""
        Computes the sum log probabilities of the labels under given logits if loss_type is not IPO, ORPO or SimPO.

        Otherwise the average log probabilities.
        """
        if self.finetuning_args.use_ref_model:
            batch = {k: v.detach().clone() for k, v in batch.items()}  # avoid error

        outputs = model(**batch, return_dict=True, use_cache=False, output_hidden_states=True)
        all_logits: "torch.Tensor" = outputs.logits.to(torch.float32)
        all_logps, valid_length = get_batch_logps(logits=all_logits, labels=batch["labels"])
        if self.loss_type in ["ipo", "orpo", "simpo"]:
            all_logps = all_logps / valid_length

        batch_size = batch["input_ids"].size(0) // 2
        chosen_logps, rejected_logps = all_logps.split(batch_size, dim=0)
        chosen_logits, rejected_logits = all_logits.split(batch_size, dim=0)
        chosen_length, _ = valid_length.split(batch_size, dim=0)

        hidden_states = outputs["hidden_states"]
        last_hidden_states = hidden_states[-1]
        hidden_states_for_prior_head = get_last_hidden_state_before_label(last_hidden_states, batch["labels"], hidden_state_offset=hidden_state_offset)

        return chosen_logps, rejected_logps, chosen_logits, rejected_logits, chosen_logps / chosen_length, hidden_states_for_prior_head

    @override
    def compute_reference_log_probs(
        self, model: "PreTrainedModel", batch: Dict[str, "torch.Tensor"]
    ) -> Tuple[Optional["torch.Tensor"], Optional["torch.Tensor"], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""
        Computes log probabilities of the reference model.
        """
        if not self.finetuning_args.use_ref_model:
            return None, None

        if self.ref_model is None:
            ref_model = model
            ref_context = self.accelerator.unwrap_model(model).disable_adapter()
        else:
            ref_model = self.ref_model
            ref_context = nullcontext()

        with torch.no_grad(), ref_context:
            reference_chosen_logps, reference_rejected_logps, reference_chosen_logits, reference_rejected_logits, _, reference_hidden_states_for_prior_head = self.concatenated_forward(ref_model, batch)

        return reference_chosen_logps, reference_rejected_logps, reference_chosen_logits, reference_rejected_logits, reference_hidden_states_for_prior_head
    
    @override
    def get_batch_loss_metrics(
        self,
        model: "PreTrainedModel",
        batch: Dict[str, "torch.Tensor"],
        train_eval: Literal["train", "eval"] = "train",
    ) -> Tuple["torch.Tensor", Dict[str, "torch.Tensor"]]:
        r"""
        Computes the DPO loss and other metrics for the given batch of inputs for train or test.
        """
        # print("input_ids[0]:", self.tokenizer.decode(batch['input_ids'][0], skip_special_tokens=True))
        # print("input_ids[1]:", self.tokenizer.decode(batch['input_ids'][1], skip_special_tokens=True))
        # print("input_ids[2]:", self.tokenizer.decode(batch['input_ids'][2], skip_special_tokens=True))
        # print("input_ids[3]:", self.tokenizer.decode(batch['input_ids'][3], skip_special_tokens=True))
        # breakpoint() # check batch
        is_gt_passage = batch.pop('is_gt_passage')


        metrics = {}
        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
            policy_chosen_logps_avg,
            policy_hidden_states_for_prior_head,
        ) = self.concatenated_forward(model, batch, hidden_state_offset=self.finetuning_args.ppl_hidden_state_offset)
        policy_prior_logits = self.prior_head(policy_hidden_states_for_prior_head)

        reference_chosen_logps, reference_rejected_logps, reference_chosen_logits, reference_rejected_logits, reference_hidden_states_for_prior_head = self.compute_reference_log_probs(model, batch)
        reference_prior_logits = self.ref_prior_head(reference_hidden_states_for_prior_head)

        losses, chosen_rewards, rejected_rewards, policy_chosen_posterior, policy_rejected_posterior = self.compute_bepo_preference_loss(
            policy_chosen_logits,
            policy_rejected_logits,
            policy_prior_logits,
            reference_chosen_logits,
            reference_rejected_logits,
            reference_prior_logits,
            batch["labels"],
        )
        # losses, chosen_rewards, rejected_rewards = self.compute_preference_loss(
        #     policy_chosen_logps,
        #     policy_rejected_logps,
        #     reference_chosen_logps,
        #     reference_rejected_logps,
        # )
        sft_loss = -policy_chosen_logps_avg
        if self.ftx_gamma > 1e-6:
            losses += self.ftx_gamma * sft_loss
        
        # Compute prior loss for the chosen examples (first half of batch)
        # Assumes idx=0 holds the GT passage in chosen examples
        K = policy_chosen_logits.shape[0]
        prior_loss = self.compute_prior_loss(
            policy_prior_logits[:K],  # Only use chosen examples for prior loss
            prior_logprob=None  # Can add softmax prior_logprob if needed
        )
        if self.finetuning_args.use_prior_head_loss:
            losses += prior_loss

        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        # DEBUG
        # print(f"chosen_rewards: {chosen_rewards.detach().mean().cpu()}, rejected_rewards: {rejected_rewards.detach().mean().cpu()}")
        # print(f"policy_chosen_posterior: {policy_chosen_posterior.detach().mean().cpu()}, policy_rejected_posterior: {policy_rejected_posterior.detach().mean().cpu()}")
        # print(f"losses: {losses.detach().mean().cpu()}")

        prefix = "eval_" if train_eval == "eval" else ""
        metrics["{}rewards/chosen".format(prefix)] = chosen_rewards.mean().cpu()
        metrics["{}rewards/rejected".format(prefix)] = rejected_rewards.mean().cpu()
        metrics["{}rewards/accuracies".format(prefix)] = reward_accuracies.mean().cpu()
        metrics["{}rewards/margins".format(prefix)] = (chosen_rewards - rejected_rewards).mean().cpu()
        metrics["{}logps/rejected".format(prefix)] = policy_rejected_logps.detach().mean().cpu()
        metrics["{}logps/chosen".format(prefix)] = policy_chosen_logps.detach().mean().cpu()
        metrics["{}logits/rejected".format(prefix)] = policy_rejected_logits.detach().mean().cpu()
        metrics["{}logits/chosen".format(prefix)] = policy_chosen_logits.detach().mean().cpu()
        metrics["{}posterior/chosen".format(prefix)] = policy_chosen_posterior.detach().mean().cpu()
        metrics["{}posterior/rejected".format(prefix)] = policy_rejected_posterior.detach().mean().cpu()
        metrics["{}prior_loss".format(prefix)] = prior_loss.detach().mean().cpu()
        
        # Compute prior head accuracy (whether GT passage at idx=0 has highest score)
        if self.finetuning_args.use_prior_head_loss:
            prior_pred = torch.argmax(policy_prior_logits[:K].squeeze(-1), dim=0)
            prior_accuracy = (prior_pred == 0).float()
            metrics["{}prior_accuracy".format(prefix)] = prior_accuracy.cpu()
        
        if self.loss_type == "orpo":
            metrics["{}sft_loss".format(prefix)] = sft_loss.detach().mean().cpu()
            metrics["{}odds_ratio_loss".format(prefix)] = ((losses - sft_loss) / self.beta).detach().mean().cpu()

        return losses.mean(), metrics
    
    @override
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """
        Override save_model to save prior_head (mlp_head) as a separate .pt file.
        """
        # Call parent save_model first
        super().save_model(output_dir, _internal_call)
        
        # Save prior_head separately if it exists (only on main process)
        if self.accelerator.is_main_process and hasattr(self, 'prior_head') and self.prior_head is not None:
            if output_dir is None:
                output_dir = self.args.output_dir
            
            # Ensure output directory exists (important for distributed training)
            os.makedirs(output_dir, exist_ok=True)
            
            prior_head_path = os.path.join(output_dir, "prior_head.pt")
            # Unwrap the model if it's wrapped by accelerator (e.g., DDP)
            prior_head_to_save = self.accelerator.unwrap_model(self.prior_head)
            torch.save(prior_head_to_save.state_dict(), prior_head_path)
            logger.info(f"Saved prior_head to {prior_head_path}")
