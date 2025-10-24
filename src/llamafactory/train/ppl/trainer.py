# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer_seq2seq.py
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

import json
import os
from types import MethodType
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from transformers import Seq2SeqTrainer
from typing_extensions import override

from ...extras.constants import IGNORE_INDEX
from ...extras.logging import get_logger
from ..callbacks import PissaConvertCallback, SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler, get_batch_logps

from collections import defaultdict
import os


if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments

from .ppl_loss import compute_ppl_loss, compute_joint_loss, compute_ensemble_loss


logger = get_logger(__name__)


def initialize_prior_head(finetuning_args: "FinetuningArguments", hidden_size: int):
    print(f"[PPL Trainer] Prior head modeling: {finetuning_args.ppl_prior_modeling}")
    print(f"[PPL Trainer] Use prior head loss: {finetuning_args.use_prior_head_loss}")
    print(f"[PPL Trainer] Hidden state offset: {finetuning_args.ppl_hidden_state_offset}")
    print(f"[PPL Trainer] Prior head loss factor: {finetuning_args.ppl_prior_loss_factor}")

    prior_head = None
    if finetuning_args.ppl_prior_modeling in ['mlp_head', 'prompted_vlm+mlp_head']:
        # Initialize a 2-layer MLP head of shape [h]
        input_dim = hidden_size
        proj_dim = finetuning_args.ppl_prior_head_proj_dim

        mlp_layers = []
        for i in range(finetuning_args.ppl_prior_head_num_of_layers - 1):
            mlp_layers.append(nn.Linear(input_dim, proj_dim))
            mlp_layers.append(nn.ReLU())
            input_dim = proj_dim
        mlp_layers.append(nn.Linear(input_dim, 1))

        prior_head = nn.Sequential(*mlp_layers)
        print(f"[PPL Trainer - Prior Head] Prior head number of layers: {finetuning_args.ppl_prior_head_num_of_layers}")
        print(f"[PPL Trainer - Prior Head] Prior head projection dimension: {proj_dim}")
        print(f"[PPL Trainer - Prior Head] Prior head parameters: {sum(p.numel() for p in prior_head.parameters())}")
        if finetuning_args.ppl_prior_head_path is not None:
            prior_head.load_state_dict(torch.load(finetuning_args.ppl_prior_head_path))
            print(f"[PPL Trainer - Prior Head] Prior head loaded from {finetuning_args.ppl_prior_head_path}")
        else:
            print(f"[PPL Trainer - Prior Head] No prior head path provided, initializing a new prior head")
    else:
        print(f"[PPL Trainer - Prior Head] No Prior head")
    return prior_head

def get_last_hidden_state_before_label(hidden_states: "torch.Tensor", labels: "torch.Tensor", hidden_state_offset: int = 0) -> "torch.Tensor":
    label_indices = (labels != IGNORE_INDEX).nonzero(as_tuple=False)
    
    # Extract the last IGNORE_INDEX index for each batch
    pre_label_indices_per_batch = [
        label_indices[label_indices[:, 0] == i, 1].min().item() - hidden_state_offset - 1
        for i in range(labels.size(0))
    ]
    pre_label_indices_tensor = torch.tensor(pre_label_indices_per_batch, device=labels.device)
    
    # Get hidden states at position just before first label
    hidden_at_pre_label = hidden_states[torch.arange(labels.size(0), device=labels.device), pre_label_indices_tensor, :]
    return hidden_at_pre_label


class CustomSeq2SeqPPLTrainer(Seq2SeqTrainer):
    r"""
    Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE.
    """

    def __init__(
        self, finetuning_args: "FinetuningArguments", processor: Optional["ProcessorMixin"], **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.finetuning_args = finetuning_args
        # Initialize metrics storage for custom logging
        self._metrics = defaultdict(list)

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.pissa_convert:
            self.add_callback(PissaConvertCallback)

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

        print(f"[PPL Trainer] Use Ensemble Loss: {finetuning_args.use_ensemble_loss}")

        print(f"[PPL Trainer] Using PPL loss type: {self.finetuning_args.ppl_loss_type}")
        print(f"[PPL Trainer] Prior head modeling: {finetuning_args.ppl_prior_modeling}")
        print(f"[PPL Trainer] Use prior head loss: {finetuning_args.use_prior_head_loss}")
        print(f"[PPL Trainer] Hidden state offset: {finetuning_args.ppl_hidden_state_offset}")
        print(f"[PPL Trainer] Prior head loss factor: {finetuning_args.ppl_prior_loss_factor}")

        # if finetuning_args.ppl_prior_modeling == 'mlp_head':
        #     # Initialize a 2-layer MLP head of shape [h]
        #     input_dim = self.model.config.hidden_size
        #     proj_dim = finetuning_args.ppl_prior_head_proj_dim

        #     mlp_layers = []
        #     for i in range(finetuning_args.ppl_prior_head_num_of_layers - 1):
        #         mlp_layers.append(nn.Linear(input_dim, proj_dim))
        #         mlp_layers.append(nn.ReLU())
        #         input_dim = proj_dim
        #     mlp_layers.append(nn.Linear(input_dim, 1))

        #     self.prior_head = nn.Sequential(*mlp_layers)
        #     print(f"[PPL Trainer - Prior Head] Prior head number of layers: {finetuning_args.ppl_prior_head_num_of_layers}")
        #     print(f"[PPL Trainer - Prior Head] Prior head projection dimension: {proj_dim}")
        #     print(f"[PPL Trainer - Prior Head] Prior head parameters: {sum(p.numel() for p in self.prior_head.parameters())}")
        #     if finetuning_args.ppl_prior_head_path is not None:
        #         self.prior_head.load_state_dict(torch.load(finetuning_args.ppl_prior_head_path))
        #         print(f"[PPL Trainer - Prior Head] Prior head loaded from {finetuning_args.ppl_prior_head_path}")
        #     else:
        #         print(f"[PPL Trainer - Prior Head] No prior head path provided, initializing a new prior head")
        #     self.prior_head.to(self.model.device)
        # else:
        #     self.prior_head = None
        #     print(f"[PPL Trainer - Prior Head] No Prior head")
        self.prior_head = initialize_prior_head(finetuning_args, hidden_size=self.model.config.hidden_size)
        self.prior_head.to(self.model.device)
        
        if self.finetuning_args.ppl_prior_loss_type in ['logistic', 'logistic+llk']:
            self.prior_loss_fn = nn.BCEWithLogitsLoss()

        print(f"[PPL Trainer - Prior Loss] Prior loss type: {self.finetuning_args.ppl_prior_loss_type}")
        
        # Freeze VLM weights if specified
        if finetuning_args.freeze_vlm_weights:
            print(f"[PPL Trainer] Freezing VLM weights...")
            for name, param in self.model.named_parameters():
                param.requires_grad = False
            
            # Unfreeze prior_head if it exists
            if self.prior_head is not None:
                for param in self.prior_head.parameters():
                    param.requires_grad = True
        else:
            print(f"[PPL Trainer] VLM weights are trainable")
        
        # if finetuning_args.use_ppl_loss:
            # print("Using PPL training with Posterior Loss.")
        # else:
            # print("Using PPL Trainer, but Posterior Loss is disabled")
            # print(f"Using PPL temperature (tau): {self.finetuning_args.ppl_temperature}")
            # print(f"Using PPL prior: {self.finetuning_args.ppl_prior}")
            # print(f"Using PPL llk factor: {self.finetuning_args.ppl_llk_factor}")
    
    def concatenated_forward(self, model, batch, hidden_state_offset=0, return_hidden_states=True):
        r"""
        The first instance of the batch is the GT passage.
        NOTE: only batch size = 1 for the collator is supported currently.
        """
        outputs = model(**batch, return_dict=True, use_cache=False, output_hidden_states=return_hidden_states)
        all_logits = outputs["logits"]
        labels = batch["labels"]
        all_logps, lengths = get_batch_logps(logits=all_logits, labels=labels)

        pos_logps, neg_logps = all_logps[:1], all_logps[1:]
        
        # Extract last-layer hidden states at position just before the first label
        hidden_at_pre_label = None
        if return_hidden_states:
            hidden_states = outputs["hidden_states"]
            last_hidden_states = hidden_states[-1]  # Shape: [batch_size, seq_len, hidden_size]
            hidden_at_pre_label = get_last_hidden_state_before_label(last_hidden_states, labels, hidden_state_offset=hidden_state_offset)
        
        return pos_logps, neg_logps, all_logits, outputs, lengths[0], hidden_at_pre_label


    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False, eval_mode=False):
        """
        Override `compute_loss` in `transformers.trainer`.
        """

        prior_inputs = None
        bs = inputs["input_ids"].size(0)
        if self.finetuning_args.ppl_prior_modeling == 'prompted_vlm+mlp_head':
            prior_inputs = {
                "input_ids": inputs["input_ids"][bs//2:],
                "attention_mask": inputs["attention_mask"][bs//2:],
                "labels": inputs["labels"][bs//2:],
                "pixel_values": inputs["pixel_values"][inputs['pixel_values'].size(0)//2:],
                "image_grid_thw": inputs["image_grid_thw"][bs//2:],
            }
            inputs = {
                "input_ids": inputs["input_ids"][:bs//2],
                "attention_mask": inputs["attention_mask"][:bs//2],
                "labels": inputs["labels"][:bs//2],
                "pixel_values": inputs["pixel_values"][:inputs['pixel_values'].size(0)//2],
                "image_grid_thw": inputs["image_grid_thw"][:bs//2],
            }

        pos_logps, neg_logps, all_logits, outputs, ans_len, hidden_at_pre_label = self.concatenated_forward(model, inputs, self.finetuning_args.ppl_hidden_state_offset, return_hidden_states=self.finetuning_args.ppl_prior_modeling == 'mlp_head')
        prior_logits = None
        if self.prior_head is not None:
            if self.finetuning_args.ppl_prior_modeling == 'mlp_head':
                prior_logits = self.prior_head(hidden_at_pre_label)  # Shape: [batch_size, 1]
            elif self.finetuning_args.ppl_prior_modeling == 'prompted_vlm+mlp_head':
                prior_outputs = model(**prior_inputs, return_dict=True, use_cache=False, output_hidden_states=True)
                prior_llk_loss = prior_outputs["loss"]
                prior_hidden_states = prior_outputs["hidden_states"]
                prior_last_hidden_states = prior_hidden_states[-1]
                hidden_states_for_prior_head = get_last_hidden_state_before_label(prior_last_hidden_states, prior_inputs["labels"], hidden_state_offset=self.finetuning_args.ppl_hidden_state_offset)
                prior_logits = self.prior_head(hidden_states_for_prior_head)
            else:
                raise NotImplementedError(f"Prior modeling type {self.finetuning_args.ppl_prior_modeling} is not implemented")

        if self.finetuning_args.ppl_loss_type == "joint":
            beft_loss, posterior_loss, llk_loss, posterior_logprob, prior_logprob = compute_joint_loss(pos_logps, all_logits, inputs["labels"], prior_logits)
        elif self.finetuning_args.ppl_loss_type == "posterior":
            beft_loss, posterior_loss, llk_loss, posterior_logprob, prior_logprob = compute_ppl_loss(pos_logps, neg_logps, prior_logits)
        elif self.finetuning_args.ppl_loss_type == "ensemble":
            beft_loss, posterior_loss, llk_loss, posterior_logprob, prior_logprob = compute_ensemble_loss(all_logits, inputs["labels"], prior_logits)
        elif self.finetuning_args.ppl_loss_type == "llk":
            beft_loss, posterior_loss, llk_loss, posterior_logprob, prior_logprob = compute_ppl_loss(pos_logps, neg_logps, prior_logits)
            beft_loss = llk_loss
        else:
            raise NotImplementedError(f"PPL loss type {self.finetuning_args.ppl_loss_type} is not implemented")
        
        loss = torch.tensor(0.0, device=self.model.device)
        if self.finetuning_args.use_ensemble_loss:
            loss = beft_loss

        prior_loss = torch.tensor(0.0)
        if self.finetuning_args.use_prior_head_loss:
            prior_lambda = None
            if self.finetuning_args.ppl_prior_loss_factor < 0:
                prior_lambda = posterior_logprob.shape[-1] # = number of answer tokens
            else:
                prior_lambda = self.finetuning_args.ppl_prior_loss_factor
            if self.finetuning_args.ppl_prior_loss_type == 'softmax':
                prior_loss = -prior_logprob[0] * prior_lambda
            elif self.finetuning_args.ppl_prior_loss_type == 'logistic':
                prior_labels = torch.zeros_like(prior_logits)
                prior_labels[0] = 1
                prior_loss = self.prior_loss_fn(prior_logits, prior_labels) * prior_lambda
            elif self.finetuning_args.ppl_prior_loss_type == 'logistic+llk':
                prior_labels = torch.zeros_like(prior_logits)
                prior_labels[0] = 1
                prior_loss = self.prior_loss_fn(prior_logits, prior_labels) * prior_lambda
                prior_loss += prior_llk_loss
            else:
                raise NotImplementedError(f"Prior loss type {self.finetuning_args.ppl_prior_loss_type} is not implemented")
            loss += prior_loss

        # Logging
        # posterior_logprob.shape = [K, # ans tokens]
        map_passage_idx = torch.argmax(posterior_logprob, dim=0) # shape (# ans tokens,)
        posterior_entropy = -torch.sum(torch.exp(posterior_logprob)*posterior_logprob, dim=0) # shape (# ans tokens,)
        prior_passage_idx = torch.argmax(prior_logprob, dim=0) # shape (# ans tokens,)
        prior_entropy = -torch.sum(torch.exp(prior_logprob)*prior_logprob, dim=0) # shape (1, )

        posterior_hitrate_over_steps = (map_passage_idx == 0).sum(-1) / map_passage_idx.shape[-1]
        prior_hitrate = (prior_passage_idx == 0)


        self._metrics["posterior_loss"].append(posterior_loss.item())
        self._metrics["llk_loss"].append(llk_loss.item())
        self._metrics["prior_loss"].append(prior_loss.item())
        self._metrics["total_loss"].append(loss.item())
        self._metrics["posterior_hit_at_first"].append(map_passage_idx[0].item() == 0)
        self._metrics["posterior_hit_at_mid"].append(map_passage_idx[ans_len//2].item() == 0)
        self._metrics["posterior_hit_at_last"].append(map_passage_idx[-1].item() == 0)
        self._metrics["posterior_hit_over_steps"].append(posterior_hitrate_over_steps.item())
        self._metrics["prior_hit"].append(prior_hitrate.item())
        self._metrics["prior_entropy"].append(prior_entropy.item())

        self._metrics["posterior_entropy_mean"].append(posterior_entropy.mean().item())
        self._metrics["posterior_entropy_at_first"].append(posterior_entropy[1].item())
        self._metrics["posterior_entropy_at_mid"].append(posterior_entropy[ans_len//2].item())
        self._metrics["posterior_entropy_at_last"].append(posterior_entropy[-1].item())

        #DEBUG
        # print(f"[PPL Trainer] Loss: {loss.item()}, Posterior Loss: {posterior_loss.item()}, LLK Loss: {llk_loss.item()}, Prior Loss: {prior_loss}")
        # print(f"[PPL Trainer] Posterior Hit (mean over steps): {posterior_hitrate_over_steps.item()}, Posterior Entropy (mean over steps): {posterior_entropy.mean().item()}")
        # print(f"[PPL Trainer] Prior Hit: {prior_hitrate.item()}")
        # print(f"[PPL Trainer] Posterior Hit (at first): {map_passage_idx[0].item() == 0}, Posterior Entropy (at first): {posterior_entropy[0].item()}")
        # print(f"[PPL Trainer] Posterior Hit (at mid): {map_passage_idx[ans_len//2].item() == 0}, Posterior Entropy (at mid): {posterior_entropy[ans_len//2].item()}")
        # print(f"[PPL Trainer] Posterior Hit (at last): {map_passage_idx[-1].item() == 0}, Posterior Entropy (at last): {posterior_entropy[-1].item()}")
        return (loss, outputs) if return_outputs else loss
    
    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: Dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""
        Removes the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        labels = inputs["labels"] if "labels" in inputs else None
        if self.args.predict_with_generate:
            assert self.tokenizer.padding_side == "left", "This method only accepts left-padded tensor."
            labels = labels.detach().clone() if labels is not None else None  # backup labels
            prompt_len, label_len = inputs["input_ids"].size(-1), inputs["labels"].size(-1)
            if prompt_len > label_len:
                inputs["labels"] = self._pad_tensors_to_target_len(inputs["labels"], inputs["input_ids"])
            if label_len > prompt_len:  # truncate the labels instead of padding the inputs (llama2 fp16 compatibility)
                inputs["labels"] = inputs["labels"][:, :prompt_len]

        loss, generated_tokens, _ = super().prediction_step(  # ignore the returned labels (may be truncated)
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, :prompt_len] = self.tokenizer.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    def _pad_tensors_to_target_len(self, src_tensor: "torch.Tensor", tgt_tensor: "torch.Tensor") -> "torch.Tensor":
        r"""
        Pads the tensor to the same length as the target tensor.
        """
        assert self.tokenizer.pad_token_id is not None, "Pad token is required."
        padded_tensor = self.tokenizer.pad_token_id * torch.ones_like(tgt_tensor)
        padded_tensor[:, -src_tensor.shape[-1] :] = src_tensor  # adopt left-padding
        return padded_tensor.contiguous()  # in contiguous memory

    def save_predictions(self, dataset: "Dataset", predict_results: "PredictionOutput") -> None:
        r"""
        Saves model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.tokenizer.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX, predict_results.predictions, self.tokenizer.pad_token_id
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.tokenizer.pad_token_id)[0]
            if len(pad_len):  # move pad token to last
                preds[i] = np.concatenate((preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1)

        decoded_inputs = self.tokenizer.batch_decode(dataset["input_ids"], skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)

        with open(output_prediction_file, "w", encoding="utf-8") as writer:
            res: List[str] = []
            for text, label, pred in zip(decoded_inputs, decoded_labels, decoded_preds):
                res.append(json.dumps({"prompt": text, "label": label, "predict": pred}, ensure_ascii=False))

            writer.write("\n".join(res))

    def log(self, logs: Dict[str, float]) -> None:
        """Override log method to include custom metrics."""
        # Calculate averaged metrics
        metrics = {}
        if self._metrics["posterior_loss"]:
            metrics["posterior_loss"] = sum(self._metrics["posterior_loss"]) / len(self._metrics["posterior_loss"])
        if self._metrics["llk_loss"]:
            metrics["llk_loss"] = sum(self._metrics["llk_loss"]) / len(self._metrics["llk_loss"])
        if self._metrics["prior_loss"]:
            metrics["prior_loss"] = sum(self._metrics["prior_loss"]) / len(self._metrics["prior_loss"])
        if self._metrics["prior_hit"]:
            metrics["prior_hit"] = sum(self._metrics["prior_hit"]) / len(self._metrics["prior_hit"])
        if self._metrics["total_loss"]:
            metrics["total_loss"] = sum(self._metrics["total_loss"]) / len(self._metrics["total_loss"])
        if self._metrics["posterior_hit_at_first"]:
            metrics["posterior_hit_at_first"] = sum(self._metrics["posterior_hit_at_first"]) / len(self._metrics["posterior_hit_at_first"])
        if self._metrics["posterior_hit_at_mid"]:
            metrics["posterior_hit_at_mid"] = sum(self._metrics["posterior_hit_at_mid"]) / len(self._metrics["posterior_hit_at_mid"])
        if self._metrics["posterior_hit_at_last"]:
            metrics["posterior_hit_at_last"] = sum(self._metrics["posterior_hit_at_last"]) / len(self._metrics["posterior_hit_at_last"])
        if self._metrics["posterior_hit_over_steps"]:
            metrics["posterior_hit_over_steps"] = sum(self._metrics["posterior_hit_over_steps"]) / len(self._metrics["posterior_hit_over_steps"])
        if self._metrics["posterior_entropy_mean"]:
            metrics["posterior_entropy_mean"] = sum(self._metrics["posterior_entropy_mean"]) / len(self._metrics["posterior_entropy_mean"])
        if self._metrics["posterior_entropy_at_first"]:
            metrics["posterior_entropy_at_first"] = sum(self._metrics["posterior_entropy_at_first"]) / len(self._metrics["posterior_entropy_at_first"])
        if self._metrics["posterior_entropy_at_mid"]:
            metrics["posterior_entropy_at_mid"] = sum(self._metrics["posterior_entropy_at_mid"]) / len(self._metrics["posterior_entropy_at_mid"])
        if self._metrics["posterior_entropy_at_last"]:
            metrics["posterior_entropy_at_last"] = sum(self._metrics["posterior_entropy_at_last"]) / len(self._metrics["posterior_entropy_at_last"])
        if self._metrics["prior_entropy"]:
            metrics["prior_entropy"] = sum(self._metrics["prior_entropy"]) / len(self._metrics["prior_entropy"])
        # Merge with existing logs
        logs = {**logs, **metrics}
        
        # Call parent log method
        super().log(logs)
        
        # Clear metrics for next cycle
        self._metrics.clear()

    @override
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """
        Override save_model to save prior_head (mlp_head) as a separate .pt file.
        """
        # Call parent save_model first
        super().save_model(output_dir, _internal_call)
        
        # Save prior_head separately if it exists
        if hasattr(self, 'prior_head') and self.prior_head is not None:
            if output_dir is None:
                output_dir = self.args.output_dir
            
            prior_head_path = os.path.join(output_dir, "prior_head.pt")
            torch.save(self.prior_head.state_dict(), prior_head_path)
            logger.info(f"Saved prior_head to {prior_head_path}")
