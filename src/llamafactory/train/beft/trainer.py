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

import os
from itertools import combinations
from typing import TYPE_CHECKING, Any, Dict, Optional

import torch
import torch.nn as nn

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
        self.beft_debug = finetuning_args.beft_debug

    def _prepare_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        inputs = super()._prepare_inputs(inputs)
        return inputs

    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False, eval_mode=False):
        is_gt_passage = inputs.pop("is_gt_passage", None)
        passage_image_paths_tokenized = inputs.pop("_passage_image_paths_tokenized", None)

        gt_subset_inputs = {}
        for key in list(inputs.keys()):
            if key.startswith("gt_subset_"):
                gt_subset_inputs[key[len("gt_subset_"):]] = inputs.pop(key)

        if is_gt_passage is not None:
            gt_passage_idx_list = [int(idx) for idx in torch.where(is_gt_passage == 1)[0].cpu().tolist()]
        else:
            gt_passage_idx_list = [0]

        K = inputs["input_ids"].size(0)

        deflection_labels = inputs.pop("deflection", None)
        if deflection_labels is not None:
            if deflection_labels.dim() == 0:
                deflection_labels = deflection_labels.unsqueeze(0)
            device = inputs["input_ids"].device
            deflection_labels = deflection_labels.to(device)
            if len(deflection_labels) == 1 and K > 1:
                deflection_labels = deflection_labels.repeat(K)
            elif len(deflection_labels) != K:
                deflection_labels = deflection_labels[0:1].repeat(K)

        return_hidden_states = self.finetuning_args.ppl_prior_modeling in ["mlp_head", "linear_head"] or self.finetuning_args.ppl_deflection_modeling in ["mlp_head", "linear_head"]
        return_hidden_states = (
            return_hidden_states
            or self.finetuning_args.ppl_prior_modeling == "dpp_mlp"
        )
        if (
            self.finetuning_args.ppl_enable_chunked_checkpoint
            and self.finetuning_args.ppl_forward_chunk_size is not None
            and K > self.finetuning_args.ppl_forward_chunk_size
        ):
            _, _, per_token_logps, all_logits, outputs, ans_len, hidden_at_pre_label, hidden_at_eos = self.concatenated_forward_chunk_checkpointing(
                model,
                inputs,
                self.finetuning_args.ppl_hidden_state_offset,
                return_hidden_states=return_hidden_states,
            )
        else:
            _, _, per_token_logps, all_logits, outputs, ans_len, hidden_at_pre_label, hidden_at_eos = self.concatenated_forward(
                model,
                inputs,
                self.finetuning_args.ppl_hidden_state_offset,
                return_hidden_states=return_hidden_states,
                return_per_token_logps=True,
            )

        prior_logits = None
        dpp_embeddings = None
        if self.prior_head is not None:
            if self.finetuning_args.ppl_prior_modeling in ["mlp_head", "linear_head"]:
                prior_logits = self.prior_head(hidden_at_pre_label)
            elif self.finetuning_args.ppl_prior_modeling == "dpp_mlp":
                prior_logits, dpp_embeddings = self.prior_head(hidden_at_pre_label)
            else:
                raise NotImplementedError(f"Prior modeling type {self.finetuning_args.ppl_prior_modeling} is not implemented")

        deflection_logits = None
        if self.deflection_head is not None and hidden_at_eos is not None:
            deflection_logits = self.deflection_head(hidden_at_eos)

        gt_subset_branch_used = False
        gt_subset_per_token_logps = None
        gt_subset_labels = None
        if (
            self.finetuning_args.beft_use_gt_subset_branch
            and len(gt_passage_idx_list) > 1
            and gt_subset_inputs.get("input_ids") is not None
        ):
            _, _, gt_subset_per_token_logps, _, _, _, _, _ = self.concatenated_forward(
                model,
                gt_subset_inputs,
                self.finetuning_args.ppl_hidden_state_offset,
                return_hidden_states=False,
                return_per_token_logps=True,
            )
            gt_subset_labels = gt_subset_inputs["labels"]
            gt_subset_branch_used = True

        from ..ppl.ppl_loss import build_dpp_kernel, compute_ensemble_loss, dpp_subset_log_prob, dpp_subset_nll

        beft_loss, posterior_loss, llk_loss, posterior_logprob, prior_logprob = compute_ensemble_loss(
            all_logits,
            inputs["labels"],
            prior_logits,
            per_token_logps=per_token_logps,
            deflection_logits=deflection_logits,
            deflection_labels=deflection_labels,
            gt_subset_per_token_logps=gt_subset_per_token_logps,
            gt_subset_labels=gt_subset_labels,
            gt_subset_loss_factor=self.finetuning_args.beft_gt_subset_loss_factor,
        )

        loss = torch.tensor(0.0, device=self.model.device)
        if self.finetuning_args.use_ensemble_loss:
            loss = beft_loss

        prior_loss = torch.tensor(0.0, device=self.model.device)
        if self.finetuning_args.use_prior_head_loss:
            prior_lambda = (
                posterior_logprob.shape[-1]
                if self.finetuning_args.ppl_prior_loss_factor < 0
                else self.finetuning_args.ppl_prior_loss_factor
            )
            if self.finetuning_args.ppl_prior_modeling == "dpp_mlp":
                if prior_logits is not None and dpp_embeddings is not None:
                    gt_indices = torch.tensor(
                        [idx for idx in gt_passage_idx_list if 0 <= idx < K],
                        device=prior_logits.device,
                        dtype=torch.long,
                    )
                    dpp_kernel = build_dpp_kernel(
                        prior_logits,
                        dpp_embeddings,
                        jitter=self.finetuning_args.ppl_dpp_jitter,
                    )
                    prior_loss = dpp_subset_nll(
                        dpp_kernel,
                        gt_indices,
                        jitter=self.finetuning_args.ppl_dpp_jitter,
                    ) * prior_lambda
                else:
                    prior_loss = torch.tensor(0.0, device=self.model.device)
            elif self.finetuning_args.ppl_prior_loss_type == "softmax":
                prior_loss = -prior_logprob[0] * prior_lambda
            elif self.finetuning_args.ppl_prior_loss_type == "logistic":
                if len(gt_passage_idx_list) > 0 and prior_logits is not None:
                    prior_labels = torch.zeros_like(prior_logits)
                    gt_indices = torch.tensor(gt_passage_idx_list, device=prior_logits.device, dtype=torch.long)
                    prior_labels[gt_indices] = 1
                    prior_loss = self.prior_loss_fn(prior_logits, prior_labels) * prior_lambda
                else:
                    prior_loss = torch.tensor(0.0, device=self.model.device)
            else:
                raise NotImplementedError(f"Prior loss type {self.finetuning_args.ppl_prior_loss_type} is not implemented")
            loss += prior_loss

        deflection_loss = torch.tensor(0.0, device=self.model.device)
        deflection_accuracy = torch.tensor(0.0, device=self.model.device)
        deflection_hit_rate = torch.tensor(0.0, device=self.model.device)
        if self.deflection_head is not None and deflection_logits is not None and deflection_labels is not None and self.finetuning_args.use_deflection_head_loss:
            import torch.nn.functional as F

            deflection_lambda = self.finetuning_args.ppl_deflection_loss_factor
            deflection_loss = F.binary_cross_entropy_with_logits(
                deflection_logits.squeeze(-1), deflection_labels.float()
            ) * deflection_lambda
            loss += deflection_loss

            deflection_probs = torch.sigmoid(deflection_logits.squeeze(-1))
            deflection_preds = (deflection_probs > 0.5).float()
            deflection_accuracy = (deflection_preds == deflection_labels.float()).float().mean()

            deflection_labels_float = deflection_labels.float()
            positive_mask = deflection_labels_float == 1.0
            if positive_mask.any():
                true_positives = (deflection_preds * deflection_labels_float).sum()
                total_positives = deflection_labels_float.sum()
                deflection_hit_rate = true_positives / total_positives
            else:
                deflection_hit_rate = torch.tensor(0.0, device=self.model.device)

        map_passage_idx = torch.argmax(posterior_logprob, dim=0)
        posterior_entropy = -torch.sum(torch.exp(posterior_logprob) * posterior_logprob, dim=0)
        prior_passage_idx = torch.argmax(prior_logprob, dim=0)
        prior_entropy = -torch.sum(torch.exp(prior_logprob) * prior_logprob, dim=0)

        posterior_hitrate_over_steps = torch.zeros(map_passage_idx.shape[-1], device=map_passage_idx.device, dtype=torch.float)
        for idx in gt_passage_idx_list:
            if 0 <= idx < K:
                posterior_hitrate_over_steps += (map_passage_idx == idx).float()
        posterior_hitrate_over_steps = posterior_hitrate_over_steps.clamp(0, 1).sum(-1) / map_passage_idx.shape[-1]

        prior_hitrate = torch.tensor(
            any(prior_passage_idx.item() == idx for idx in gt_passage_idx_list),
            device=prior_passage_idx.device,
            dtype=torch.float,
        )

        if self.finetuning_args.ppl_prior_modeling == "dpp_mlp" and prior_logits is not None and dpp_embeddings is not None:
            dpp_kernel = build_dpp_kernel(
                prior_logits,
                dpp_embeddings,
                jitter=self.finetuning_args.ppl_dpp_jitter,
            )
            gt_tuple = tuple(sorted(idx for idx in gt_passage_idx_list if 0 <= idx < K))
            gt_size = len(gt_tuple)

            if self.finetuning_args.ppl_dpp_candidate_mode == "all_if_small_k" and K <= self.finetuning_args.ppl_dpp_all_if_small_k_max_k:
                candidate_subsets = [
                    subset
                    for r in range(0, K + 1)
                    for subset in combinations(range(K), r)
                ]
            else:
                if gt_size == 0:
                    candidate_subsets = [tuple()]
                else:
                    candidate_subsets = list(combinations(range(K), gt_size))

            if gt_tuple not in candidate_subsets:
                candidate_subsets.append(gt_tuple)

            best_score = None
            best_subset = None
            for subset in candidate_subsets:
                subset_idx = torch.tensor(subset, device=dpp_kernel.device, dtype=torch.long)
                score = dpp_subset_log_prob(dpp_kernel, subset_idx, jitter=self.finetuning_args.ppl_dpp_jitter)
                if best_score is None or score > best_score:
                    best_score = score
                    best_subset = subset

            prior_hitrate = torch.tensor(
                1.0 if best_subset == gt_tuple else 0.0,
                device=dpp_kernel.device,
                dtype=torch.float,
            )

        prior_accuracy_threshold_0_5 = torch.tensor(1.0, device=self.model.device, dtype=torch.float)
        if prior_logits is not None and len(gt_passage_idx_list) > 0:
            prior_probs = torch.sigmoid(prior_logits.squeeze(-1))
            all_gt_above_threshold = all(
                (0 <= idx < prior_probs.size(0)) and (prior_probs[idx] > 0.5)
                for idx in gt_passage_idx_list
            )
            prior_accuracy_threshold_0_5 = torch.tensor(
                1.0 if all_gt_above_threshold else 0.0,
                device=prior_logits.device,
                dtype=torch.float,
            )

        self._metrics["posterior_loss"].append(posterior_loss.item())
        self._metrics["llk_loss"].append(llk_loss.item())
        self._metrics["prior_loss"].append(prior_loss.item())
        self._metrics["deflection_loss"].append(deflection_loss.item())
        self._metrics["deflection_accuracy"].append(deflection_accuracy.item())
        self._metrics["deflection_hit_rate"].append(deflection_hit_rate.item())
        self._metrics["total_loss"].append(loss.item())
        self._metrics["posterior_hit_at_first"].append(any(map_passage_idx[0].item() == idx for idx in gt_passage_idx_list))
        self._metrics["posterior_hit_at_mid"].append(any(map_passage_idx[ans_len // 2].item() == idx for idx in gt_passage_idx_list))
        self._metrics["posterior_hit_at_last"].append(any(map_passage_idx[-1].item() == idx for idx in gt_passage_idx_list))
        self._metrics["posterior_hit_over_steps"].append(posterior_hitrate_over_steps.item())
        self._metrics["prior_hit"].append(prior_hitrate.item())
        self._metrics["prior_accuracy_threshold_0.5"].append(prior_accuracy_threshold_0_5.item())
        self._metrics["prior_entropy"].append(prior_entropy.item())
        self._metrics["gt_subset_branch_used"].append(float(gt_subset_branch_used))
        self._metrics["num_gt_docs"].append(float(len(gt_passage_idx_list)))
        self._metrics["posterior_entropy_mean"].append(posterior_entropy.mean().item())
        self._metrics["posterior_entropy_at_first"].append(posterior_entropy[1].item())
        self._metrics["posterior_entropy_at_mid"].append(posterior_entropy[ans_len // 2].item())
        self._metrics["posterior_entropy_at_last"].append(posterior_entropy[-1].item())

        if self.beft_debug:
            print("\n" + "=" * 80)
            print("BEFT DEBUG: Batch Information")
            print("=" * 80)

            passage_image_paths = None
            if passage_image_paths_tokenized is not None:
                if isinstance(passage_image_paths_tokenized, torch.Tensor):
                    path_bytes = bytes(passage_image_paths_tokenized.cpu().tolist())
                else:
                    path_bytes = bytes(passage_image_paths_tokenized)
                all_paths_str = path_bytes.decode("utf-8")

                encoded_paths = all_paths_str.split(":::")
                passage_image_paths = []
                for encoded_path in encoded_paths:
                    if encoded_path:
                        individual_paths = [p for p in encoded_path.split("|||") if p]
                        passage_image_paths.append(individual_paths)
                    else:
                        passage_image_paths.append([])
                while len(passage_image_paths) < K:
                    passage_image_paths.append([])

            print("\n[1] Passages with Question, Answer, and Images:")
            print("-" * 80)
            for i in range(K):
                input_ids = inputs["input_ids"][i]
                labels = inputs.get("labels", None)

                non_pad_mask = input_ids != self.tokenizer.pad_token_id
                input_ids_clean = input_ids[non_pad_mask] if non_pad_mask.any() else input_ids
                full_text = self.tokenizer.decode(input_ids_clean, skip_special_tokens=False)

                answer = ""
                if labels is not None:
                    labels_i = labels[i]
                    non_ignore_mask = labels_i != -100
                    if non_ignore_mask.any():
                        labels_clean = labels_i[non_ignore_mask]
                        answer = self.tokenizer.decode(labels_clean, skip_special_tokens=True)

                question = full_text
                if answer and answer in question:
                    question = question.replace(answer, "").strip()

                is_gt = i in gt_passage_idx_list
                gt_status = "✓ GT PASSAGE" if is_gt else "✗ Not GT"

                print(f"\nPassage {i} [{gt_status}]:")
                print(f"  Question: {question}")
                print(f"  Answer: {answer}")

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
                            print("  Images: None (empty list)")
                    else:
                        print(f"  Images: Not available (index {i} >= {len(passage_image_paths)})")
                else:
                    print("  Images: Not available (passage_image_paths is None)")

            print("\n[2] GT Passage Summary:")
            print("-" * 80)
            print(f"GT passage indices: {gt_passage_idx_list}")
            print(f"Total passages: {K}")
            print(f"GT passages: {len(gt_passage_idx_list)}")
            print(f"Negative passages: {K - len(gt_passage_idx_list)}")

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

            print("\n" + "=" * 80)
            print("DEBUG: Breakpoint - Press 'c' to continue or inspect variables")
            print("=" * 80 + "\n")
            breakpoint()

        return (loss, outputs) if return_outputs else loss

    def log(self, logs: Dict[str, float], *args, **kwargs) -> None:
        metrics = {}
        if self._metrics["posterior_loss"]:
            metrics["posterior_loss"] = sum(self._metrics["posterior_loss"]) / len(self._metrics["posterior_loss"])
        if self._metrics["llk_loss"]:
            metrics["llk_loss"] = sum(self._metrics["llk_loss"]) / len(self._metrics["llk_loss"])
        if self._metrics["prior_loss"]:
            metrics["prior_loss"] = sum(self._metrics["prior_loss"]) / len(self._metrics["prior_loss"])
        if self._metrics.get("deflection_loss"):
            metrics["deflection_loss"] = sum(self._metrics["deflection_loss"]) / len(self._metrics["deflection_loss"])
        if self._metrics.get("deflection_accuracy"):
            metrics["deflection_accuracy"] = sum(self._metrics["deflection_accuracy"]) / len(self._metrics["deflection_accuracy"])
        if self._metrics.get("deflection_hit_rate"):
            metrics["deflection_hit_rate"] = sum(self._metrics["deflection_hit_rate"]) / len(self._metrics["deflection_hit_rate"])
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
        if self._metrics.get("gt_subset_branch_used"):
            metrics["gt_subset_branch_used"] = sum(self._metrics["gt_subset_branch_used"]) / len(self._metrics["gt_subset_branch_used"])
        if self._metrics.get("num_gt_docs"):
            metrics["num_gt_docs"] = sum(self._metrics["num_gt_docs"]) / len(self._metrics["num_gt_docs"])

        logs = {**logs, **metrics}
        super().log(logs, *args, **kwargs)
        self._metrics.clear()


__all__ = ["CustomSeq2SeqBEFTTrainer"]
