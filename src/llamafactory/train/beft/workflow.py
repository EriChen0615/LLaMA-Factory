# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/summarization/run_summarization.py
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

from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch

from ...data import get_dataset, get_template_and_fix_tokenizer, MultiModalDataCollatorForSeq2Seq
from ...extras.constants import IGNORE_INDEX
from ...extras.misc import get_logits_processor
from ...extras.ploting import plot_loss
from ...model import load_model, load_tokenizer
from ..trainer_utils import create_modelcard_and_push
from .trainer import CustomSeq2SeqBEFTTrainer
from ..ppl.metric import ComputeAccuracy, ComputeSimilarity, eval_logit_processor
from dataclasses import dataclass


if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments



@dataclass
class BEFTDataCollator(MultiModalDataCollatorForSeq2Seq):
    """
    BEFT Data Collator - Similar to PPLDataCollator but handles passage-specific images.
    Each passage has its own images list stored in all_passage_images.
    BEFT does NOT perform swap operation - uses original gt_passage_idx.
    """

    def _build_gt_subset_batch(self, features: List[Dict[str, Any]]) -> Optional[Dict[str, torch.Tensor]]:
        gt_subset_features = []
        for feature in features:
            if feature.get("gt_subset_input_ids") is None:
                continue

            gt_subset_features.append({
                "input_ids": feature["gt_subset_input_ids"],
                "attention_mask": feature["gt_subset_attention_mask"],
                "labels": feature["gt_subset_labels"],
                "images": feature.get("gt_subset_images") or feature.get("images"),
                "videos": feature["videos"],
            })

        if not gt_subset_features:
            return None

        gt_subset_batch = super().__call__(gt_subset_features)
        return {f"gt_subset_{key}": value for key, value in gt_subset_batch.items()}

    def __call__(self, features):
        gt_subset_batch = self._build_gt_subset_batch(features)
        concatenated_features = []
        passage_image_paths_batch = []
        deflection_labels = []

        for feature in features:
            K = len(feature["all_input_ids"])
            expanded_features = [None] * K
            gt_passage_idx = feature["gt_passage_idx"]
            deflection_label = feature.get("deflection", 0)
            deflection_labels.append(deflection_label)

            if isinstance(gt_passage_idx, list):
                gt_passage_idx_set = {int(idx) for idx in gt_passage_idx if int(idx) != -1}
            else:
                gt_passage_idx_set = {int(gt_passage_idx)} if gt_passage_idx != -1 else set()

            all_passage_images = feature.get("all_passage_images", None)
            feature_image_paths = []

            for idx, (input_ids, attention_mask, labels) in enumerate(
                zip(feature["all_input_ids"], feature["all_attention_mask"], feature["all_labels"])
            ):
                passage_images = all_passage_images[idx] if idx < len(all_passage_images) else feature["images"]
                passage_image_paths = []
                if isinstance(passage_images, list):
                    passage_image_paths = [img for img in passage_images if isinstance(img, str)]
                elif isinstance(passage_images, str):
                    passage_image_paths = [passage_images]
                feature_image_paths.append(passage_image_paths)

                expanded_features[idx] = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                    "images": passage_images,
                    "videos": feature["videos"],
                    "is_gt_passage": int(idx in gt_passage_idx_set),
                }
            concatenated_features.extend(expanded_features)
            passage_image_paths_batch.append(feature_image_paths)

        batch = super().__call__(concatenated_features)

        if len(passage_image_paths_batch) > 0:
            if len(passage_image_paths_batch) == 1:
                all_paths = passage_image_paths_batch[0]
            else:
                all_paths = []
                for feature_paths in passage_image_paths_batch:
                    all_paths.extend(feature_paths)

            encoded_paths = []
            for passage_paths in all_paths:
                if isinstance(passage_paths, list):
                    valid_paths = [str(p) for p in passage_paths if p]
                    passage_str = "|||".join(valid_paths)
                else:
                    passage_str = str(passage_paths) if passage_paths else ""
                encoded_paths.append(passage_str)

            all_paths_str = ":::".join(encoded_paths)
            path_bytes = all_paths_str.encode("utf-8")
            batch["_passage_image_paths_tokenized"] = torch.tensor(list(path_bytes), dtype=torch.long)

        if deflection_labels:
            batch["deflection"] = torch.tensor(deflection_labels, dtype=torch.long)

        if gt_subset_batch is not None:
            batch.update(gt_subset_batch)

        return batch

def run_beft(

    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[List["TrainerCallback"]] = None,
):
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    # ENFORCE LEFT PADDING
    tokenizer.padding_side = "left"
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage=finetuning_args.stage, **tokenizer_module)
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)

    if getattr(model, "is_quantized", False) and not training_args.do_train:
        setattr(model, "_hf_peft_config_loaded", True)  # hack here: make model compatible with prediction

    data_collator = BEFTDataCollator(
        template=template,
        pad_to_multiple_of=8 if training_args.do_train else None,  # for shift short attention
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
        **tokenizer_module,
    )

    # Override the decoding parameters of Seq2SeqTrainer
    training_args.generation_max_length = training_args.generation_max_length or data_args.cutoff_len
    training_args.generation_num_beams = data_args.eval_num_beams or training_args.generation_num_beams
    training_args.remove_unused_columns = False  # important for multimodal dataset

    # Metric utils
    metric_module = {}
    if training_args.predict_with_generate:
        metric_module["compute_metrics"] = ComputeSimilarity(tokenizer=tokenizer)
    elif finetuning_args.compute_accuracy:
        metric_module["compute_metrics"] = ComputeAccuracy()
        metric_module["preprocess_logits_for_metrics"] = eval_logit_processor

    # Initialize our Trainer
    trainer = CustomSeq2SeqBEFTTrainer(
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        data_collator=data_collator,
        callbacks=callbacks,
        **dataset_module,
        **tokenizer_module,
        **metric_module,
    )

    # Keyword arguments for `model.generate`
    gen_kwargs = generating_args.to_dict()
    gen_kwargs["eos_token_id"] = [tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids
    gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
    gen_kwargs["logits_processor"] = get_logits_processor()

    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss", "eval_loss", "eval_accuracy"])

    if training_args.predict_with_generate:
        tokenizer.padding_side = "left"  # use left-padding in generation

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval", **gen_kwargs)
        if training_args.predict_with_generate:  # eval_loss will be wrong if predict_with_generate is enabled
            metrics.pop("eval_loss", None)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Predict
    if training_args.do_predict:
        predict_results = trainer.predict(dataset_module["eval_dataset"], metric_key_prefix="predict", **gen_kwargs)
        if training_args.predict_with_generate:  # predict_loss will be wrong if predict_with_generate is enabled
            predict_results.metrics.pop("predict_loss", None)
        trainer.log_metrics("predict", predict_results.metrics)
        trainer.save_metrics("predict", predict_results.metrics)
        trainer.save_predictions(dataset_module["eval_dataset"], predict_results)

    # Create model card
    create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)

