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

"""
BEFT Trainer - Supports multiple gt_passage_idx (list format).
The main difference from PPL trainer is:
1. No swap operation in data collator - uses original gt_passage_idx
2. Logistic loss: all gt_passage_idx passages are treated as positive examples
3. Prior accuracy: new metric to check if all gt_passage_idx passages exceed 0.5 threshold
"""

import torch
import torch.nn as nn
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
from typing_extensions import override
from PIL import Image

from ..ppl.trainer import CustomSeq2SeqPPLTrainer

if TYPE_CHECKING:
    from ...hparams import FinetuningArguments


class CustomSeq2SeqBEFTTrainer(CustomSeq2SeqPPLTrainer):
    r"""
    BEFT Trainer that supports multiple gt_passage_idx (list format).
    Inherits from CustomSeq2SeqPPLTrainer and overrides compute_loss to handle multiple GT passages.
    """
    
    def __init__(self, finetuning_args: "FinetuningArguments", processor: Optional["ProcessorMixin"], **kwargs) -> None:
        super().__init__(finetuning_args, processor, **kwargs)
        # Enable debug mode if beft_debug is set in finetuning_args
        self.beft_debug = finetuning_args.beft_debug
    
    def _prepare_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Override _prepare_inputs to handle _passage_image_paths_tokenized.
        The tokenized paths are already a tensor, so they can be moved to device safely.
        """
        # _passage_image_paths_tokenized is already a tensor, so it will be handled by parent
        inputs = super()._prepare_inputs(inputs)
        return inputs

    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False, eval_mode=False):
        """
        Override `compute_loss` in `CustomSeq2SeqPPLTrainer`.
        Main differences:
        1. Handles gt_passage_idx as list (multiple GT passages)
        2. Logistic loss: all gt_passage_idx passages are positive examples
        3. Prior accuracy: new metric to check if all gt_passage_idx passages exceed 0.5 threshold
        """
        # Extract is_gt_passage from inputs and rebuild gt_passage_idx_list
        is_gt_passage = inputs.pop("is_gt_passage", None)
        
        # Extract _passage_image_paths_tokenized before calling model (model doesn't accept this argument)
        passage_image_paths_tokenized = inputs.pop("_passage_image_paths_tokenized", None)
        
        # Rebuild gt_passage_idx_list from is_gt_passage flags
        if is_gt_passage is not None:
            # is_gt_passage is a tensor of shape [K] with 1 for GT passages, 0 otherwise
            gt_passage_idx_list = [int(idx) for idx in torch.where(is_gt_passage == 1)[0].cpu().tolist()]
        else:
            gt_passage_idx_list = [0]  # Fallback
        
        bs = inputs["input_ids"].size(0)
        K = bs

        # Use chunked forward with checkpointing if K exceeds chunk size
        if self.finetuning_args.ppl_enable_chunked_checkpoint and self.finetuning_args.ppl_forward_chunk_size is not None and K > self.finetuning_args.ppl_forward_chunk_size:
            pos_logps, neg_logps, per_token_logps, all_logits, outputs, ans_len, hidden_at_pre_label = self.concatenated_forward_chunk_checkpointing(model, inputs, self.finetuning_args.ppl_hidden_state_offset, return_hidden_states=self.finetuning_args.ppl_prior_modeling in ['mlp_head', 'linear_head'])
        else:
            pos_logps, neg_logps, _, all_logits, outputs, ans_len, hidden_at_pre_label = self.concatenated_forward(model, inputs, self.finetuning_args.ppl_hidden_state_offset, return_hidden_states=self.finetuning_args.ppl_prior_modeling in ['mlp_head', 'linear_head'])
            per_token_logps = None

        prior_logits = None
        if self.prior_head is not None:
            if self.finetuning_args.ppl_prior_modeling in ['mlp_head', 'linear_head']:
                prior_logits = self.prior_head(hidden_at_pre_label)  # Shape: [batch_size, 1]
            else:
                raise NotImplementedError(f"Prior modeling type {self.finetuning_args.ppl_prior_modeling} is not implemented")

        from ..ppl.ppl_loss import compute_ensemble_loss
        
        # Only use ensemble loss
        beft_loss, posterior_loss, llk_loss, posterior_logprob, prior_logprob = compute_ensemble_loss(all_logits, inputs["labels"], prior_logits, per_token_logps=per_token_logps)
        
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
                # BEFT: All gt_passage_idx passages are positive examples
                if len(gt_passage_idx_list) > 0:
                    prior_labels = torch.zeros_like(prior_logits)
                    gt_indices = torch.tensor(gt_passage_idx_list, device=prior_logits.device, dtype=torch.long)
                    prior_labels[gt_indices] = 1
                    prior_loss = self.prior_loss_fn(prior_logits, prior_labels) * prior_lambda
                else:
                    # No GT passages (e.g., augmented data), skip logistic loss
                    prior_loss = torch.tensor(0.0, device=prior_logits.device)
            else:
                raise NotImplementedError(f"Prior loss type {self.finetuning_args.ppl_prior_loss_type} is not implemented")
            loss += prior_loss

        # Logging
        # posterior_logprob.shape = [K, # ans tokens]
        map_passage_idx = torch.argmax(posterior_logprob, dim=0) # shape (# ans tokens,)
        posterior_entropy = -torch.sum(torch.exp(posterior_logprob)*posterior_logprob, dim=0) # shape (# ans tokens,)
        prior_passage_idx = torch.argmax(prior_logprob, dim=0) # shape (# ans tokens,)
        prior_entropy = -torch.sum(torch.exp(prior_logprob)*prior_logprob, dim=0) # shape (1, )

        # Posterior hit rate: check if any gt_passage_idx is selected at each step
        posterior_hitrate_over_steps = torch.zeros(map_passage_idx.shape[-1], device=map_passage_idx.device, dtype=torch.float)
        for idx in gt_passage_idx_list:
            if 0 <= idx < K:
                posterior_hitrate_over_steps += (map_passage_idx == idx).float()
        posterior_hitrate_over_steps = posterior_hitrate_over_steps.clamp(0, 1).sum(-1) / map_passage_idx.shape[-1]
        
        # Prior hit rate: check if top prior passage is in gt_passage_idx_list
        prior_hitrate = torch.tensor(any(prior_passage_idx.item() == idx for idx in gt_passage_idx_list), device=prior_passage_idx.device, dtype=torch.float)
        
        # BEFT: Prior accuracy with 0.5 threshold - check if all gt_passage_idx passages exceed 0.5
        prior_accuracy_threshold_0_5 = torch.tensor(1.0, device=prior_logits.device, dtype=torch.float)
        if prior_logits is not None and len(gt_passage_idx_list) > 0:
            prior_probs = torch.sigmoid(prior_logits.squeeze(-1))  # Shape: [K]
            all_gt_above_threshold = all(
                (0 <= idx < prior_probs.size(0)) and (prior_probs[idx] > 0.5)
                for idx in gt_passage_idx_list
            )
            prior_accuracy_threshold_0_5 = torch.tensor(1.0 if all_gt_above_threshold else 0.0, device=prior_logits.device, dtype=torch.float)
        
        self._metrics["posterior_loss"].append(posterior_loss.item())
        self._metrics["llk_loss"].append(llk_loss.item())
        self._metrics["prior_loss"].append(prior_loss.item())
        self._metrics["total_loss"].append(loss.item())
        self._metrics["posterior_hit_at_first"].append(any(map_passage_idx[0].item() == idx for idx in gt_passage_idx_list))
        self._metrics["posterior_hit_at_mid"].append(any(map_passage_idx[ans_len//2].item() == idx for idx in gt_passage_idx_list))
        self._metrics["posterior_hit_at_last"].append(any(map_passage_idx[-1].item() == idx for idx in gt_passage_idx_list))
        self._metrics["posterior_hit_over_steps"].append(posterior_hitrate_over_steps.item())
        self._metrics["prior_hit"].append(prior_hitrate.item())
        self._metrics["prior_accuracy_threshold_0.5"].append(prior_accuracy_threshold_0_5.item())
        self._metrics["prior_entropy"].append(prior_entropy.item())

        self._metrics["posterior_entropy_mean"].append(posterior_entropy.mean().item())
        self._metrics["posterior_entropy_at_first"].append(posterior_entropy[1].item())
        self._metrics["posterior_entropy_at_mid"].append(posterior_entropy[ans_len//2].item())
        self._metrics["posterior_entropy_at_last"].append(posterior_entropy[-1].item())

        # ========== DEBUG STATEMENTS ==========
        if self.beft_debug:
            print("\n" + "="*80)
            print("BEFT DEBUG: Batch Information")
            print("="*80)
            
            # Decode image paths first
            passage_image_paths = None
            if passage_image_paths_tokenized is not None:
                # Convert tensor back to bytes and then to string
                if isinstance(passage_image_paths_tokenized, torch.Tensor):
                    path_bytes = bytes(passage_image_paths_tokenized.cpu().tolist())
                else:
                    path_bytes = bytes(passage_image_paths_tokenized)
                all_paths_str = path_bytes.decode('utf-8')
                
                # Decode: split by ::: to get passages, then by ||| to get individual paths
                encoded_paths = all_paths_str.split(":::")
                passage_image_paths = []
                for encoded_path in encoded_paths:
                    if encoded_path:
                        # Split by ||| to get individual image paths
                        individual_paths = encoded_path.split("|||")
                        # Filter out empty strings
                        individual_paths = [p for p in individual_paths if p]
                        passage_image_paths.append(individual_paths)
                    else:
                        passage_image_paths.append([])
                
                # Ensure we have paths for all K passages (pad with empty lists if needed)
                while len(passage_image_paths) < K:
                    passage_image_paths.append([])
            
            # 1. Print Question, Answer, and Images for each passage
            print("\n[1] Passages with Question, Answer, and Images:")
            print("-" * 80)
            for i in range(K):
                input_ids = inputs["input_ids"][i]
                labels = inputs.get("labels", None)
                
                # Remove padding tokens
                non_pad_mask = input_ids != self.tokenizer.pad_token_id
                if non_pad_mask.any():
                    input_ids_clean = input_ids[non_pad_mask]
                else:
                    input_ids_clean = input_ids
                
                # Decode full input to extract question and answer
                full_text = self.tokenizer.decode(input_ids_clean, skip_special_tokens=False)
                
                # Extract answer from labels
                answer = ""
                if labels is not None:
                    labels_i = labels[i]
                    non_ignore_mask = labels_i != -100
                    if non_ignore_mask.any():
                        labels_clean = labels_i[non_ignore_mask]
                        answer = self.tokenizer.decode(labels_clean, skip_special_tokens=True)
                
                # Try to extract question from full text (remove answer part if found)
                # For Qwen2-VL format, question is usually before the answer
                question = full_text
                if answer and answer in question:
                    # Remove answer from question
                    question = question.replace(answer, "").strip()
                
                # Check if this is a GT passage
                is_gt = i in gt_passage_idx_list
                gt_status = "✓ GT PASSAGE" if is_gt else "✗ Not GT"
                
                # Print passage information
                print(f"\nPassage {i} [{gt_status}]:")
                print(f"  Question: {question}")
                print(f"  Answer: {answer}")
                
                # Print image paths (always show, even if empty)
                if passage_image_paths is not None:
                    if i < len(passage_image_paths):
                        passage_images = passage_image_paths[i]
                        if isinstance(passage_images, list) and len(passage_images) > 0:
                            print(f"  Images ({len(passage_images)}):")
                            for img_idx, orig_path in enumerate(passage_images):
                                if orig_path and isinstance(orig_path, str):
                                    exists = "✓" if os.path.exists(orig_path) else "✗"
                                    print(f"    [{img_idx}] {exists} {orig_path}")
                                else:
                                    print(f"    [{img_idx}] Invalid path: {orig_path}")
                        else:
                            print(f"  Images: None (empty list)")
                    else:
                        print(f"  Images: Not available (index {i} >= {len(passage_image_paths)})")
                else:
                    print(f"  Images: Not available (passage_image_paths is None)")
            
            # 2. Summary of GT passages
            print("\n[2] GT Passage Summary:")
            print("-" * 80)
            print(f"GT passage indices: {gt_passage_idx_list}")
            print(f"Total passages: {K}")
            print(f"GT passages: {len(gt_passage_idx_list)}")
            print(f"Negative passages: {K - len(gt_passage_idx_list)}")
            
            # 3. Image paths summary
            if passage_image_paths is not None:
                print("\n[3] Image Paths Summary:")
                print("-" * 80)
                print(f"Total decoded passage paths: {len(passage_image_paths)}")
                for i in range(K):
                    if i < len(passage_image_paths):
                        path_count = len(passage_image_paths[i]) if isinstance(passage_image_paths[i], list) else 0
                        print(f"  Passage {i}: {path_count} image(s)")
                    else:
                        print(f"  Passage {i}: No paths decoded")
            
            print("\n" + "="*80)
            print("DEBUG: Breakpoint - Press 'c' to continue or inspect variables")
            print("="*80 + "\n")
            breakpoint()
        # ========== END DEBUG STATEMENTS ==========

        return (loss, outputs) if return_outputs else loss

    def log(self, logs: Dict[str, float]) -> None:
        """Override log method to include custom metrics including prior_accuracy_threshold_0.5."""
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
        if self._metrics.get("prior_accuracy_threshold_0.5"):
            metrics["prior_accuracy_threshold_0.5"] = sum(self._metrics["prior_accuracy_threshold_0.5"]) / len(self._metrics["prior_accuracy_threshold_0.5"])
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

__all__ = ["CustomSeq2SeqBEFTTrainer"]
