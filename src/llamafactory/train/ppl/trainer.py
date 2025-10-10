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
from transformers import Seq2SeqTrainer
from typing_extensions import override

from ...extras.constants import IGNORE_INDEX
from ...extras.logging import get_logger
from ..callbacks import PissaConvertCallback, SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler, get_batch_logps

from collections import defaultdict


if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments

from .ppl_loss import compute_ppl_loss, compute_joint_loss, compute_ensemble_loss


logger = get_logger(__name__)


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

        print(f"[PPL Trainer] Using PPL loss type: {self.finetuning_args.ppl_loss_type}")
        # if finetuning_args.use_ppl_loss:
            # print("Using PPL training with Posterior Loss.")
        # else:
            # print("Using PPL Trainer, but Posterior Loss is disabled")
            # print(f"Using PPL temperature (tau): {self.finetuning_args.ppl_temperature}")
            # print(f"Using PPL prior: {self.finetuning_args.ppl_prior}")
            # print(f"Using PPL llk factor: {self.finetuning_args.ppl_llk_factor}")
    
    def concatenated_forward(self, model, batch):
        r"""
        The first instance of the batch is the GT passage.
        NOTE: only batch size = 1 for the collator is supported currently.
        """
        outputs = model(**batch, return_dict=True, use_cache=False)
        all_logits = outputs["logits"]
        all_logps, lengths = get_batch_logps(logits=all_logits, labels=batch["labels"])

        pos_logps, neg_logps = all_logps[:1], all_logps[1:]
        return pos_logps, neg_logps, all_logits, outputs, lengths[0]


    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False, eval_mode=False):
        """
        Override `compute_loss` in `transformers.trainer`. Below is the original code. 
        """

        pos_logps, neg_logps, all_logits, outputs, ans_len = self.concatenated_forward(model, inputs)

        if self.finetuning_args.ppl_loss_type == "joint":
            loss, posterior_loss, llk_loss, posterior_logprob = compute_joint_loss(pos_logps, all_logits, inputs["labels"])
        elif self.finetuning_args.ppl_loss_type == "posterior":
            loss, posterior_loss, llk_loss, posterior_logprob = compute_ppl_loss(pos_logps, neg_logps)
        elif self.finetuning_args.ppl_loss_type == "ensemble":
            loss, posterior_loss, llk_loss, posterior_logprob = compute_ensemble_loss(all_logits, inputs["labels"])
        elif self.finetuning_args.ppl_loss_type == "llk":
            loss, posterior_loss, llk_loss, posterior_logprob = compute_ppl_loss(pos_logps, neg_logps)
            loss = llk_loss
        else:
            raise NotImplementedError(f"PPL loss type {self.finetuning_args.ppl_loss_type} is not implemented")

        # Logging
        # posterior_logprob.shape = [K, # ans tokens]
        map_passage_idx = torch.argmax(posterior_logprob, dim=0) # shape (# ans tokens,)
        posterior_entropy = -torch.sum(torch.exp(posterior_logprob)*posterior_logprob, dim=0) # shape (# ans tokens,)

        posterior_hitrate_over_steps = (map_passage_idx == 0).sum(-1) / map_passage_idx.shape[-1]


        self._metrics["posterior_loss"].append(posterior_loss.item())
        self._metrics["llk_loss"].append(llk_loss.item())
        self._metrics["total_loss"].append(loss.item())
        self._metrics["posterior_hit_at_first"].append(map_passage_idx[0].item() == 0)
        self._metrics["posterior_hit_at_mid"].append(map_passage_idx[ans_len//2].item() == 0)
        self._metrics["posterior_hit_at_last"].append(map_passage_idx[-1].item() == 0)
        self._metrics["posterior_hit_over_steps"].append(posterior_hitrate_over_steps.item())

        self._metrics["posterior_entropy_mean"].append(posterior_entropy.mean().item())
        self._metrics["posterior_entropy_at_first"].append(posterior_entropy[1].item())
        self._metrics["posterior_entropy_at_mid"].append(posterior_entropy[ans_len//2].item())
        self._metrics["posterior_entropy_at_last"].append(posterior_entropy[-1].item())

        #DEBUG
        # print(f"[PPL Trainer] Loss: {loss.item()}, Posterior Loss: {posterior_loss.item()}, LLK Loss: {llk_loss.item()}")
        # print(f"[PPL Trainer] Posterior Hit (mean over steps): {posterior_hitrate_over_steps.item()}, Posterior Entropy (mean over steps): {posterior_entropy.mean().item()}")
        # print(f"[PPL Trainer] Posterior Hit (at first): {map_passage_idx[0].item() == 0}, Posterior Entropy (at first): {posterior_entropy[0].item()}")
        # print(f"[PPL Trainer] Posterior Hit (at mid): {map_passage_idx[ans_len//2].item() == 0}, Posterior Entropy (at mid): {posterior_entropy[ans_len//2].item()}")
        # print(f"[PPL Trainer] Posterior Hit (at last): {map_passage_idx[-1].item() == 0}, Posterior Entropy (at last): {posterior_entropy[-1].item()}")
        # breakpoint()
 #
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
        # Merge with existing logs
        logs = {**logs, **metrics}
        
        # Call parent log method
        super().log(logs)
        
        # Clear metrics for next cycle
        self._metrics.clear()
