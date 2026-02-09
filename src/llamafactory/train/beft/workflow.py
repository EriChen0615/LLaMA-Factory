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

from typing import TYPE_CHECKING, List, Optional

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
    def __call__(self, features):
        concatenated_features = []
        passage_image_paths_batch = []  # Store image paths for each feature
        deflection_labels = []  # Store deflection labels (one per feature)
        
        for feature in features:
            K = len(feature["all_input_ids"])
            expanded_features = [None] * K
            gt_passage_idx = feature["gt_passage_idx"]
            # Extract deflection label (default to 0 if not present)
            deflection_label = feature.get("deflection", 0)
            deflection_labels.append(deflection_label)
            
            # Normalize gt_passage_idx to list format
            if isinstance(gt_passage_idx, list):
                # Filter out -1 values (used as placeholder for "no GT")
                gt_passage_idx_set = {int(idx) for idx in gt_passage_idx if int(idx) != -1}
            else:
                gt_passage_idx_set = {int(gt_passage_idx)} if gt_passage_idx != -1 else set()
            
            # Get passage-specific images if available, otherwise use main images
            all_passage_images = feature.get("all_passage_images", None)
            # Store image paths for this feature (for debugging)
            feature_image_paths = []
            
            # BEFT: No swap operation - keep original order
            # Add is_gt_passage flag to each expanded feature (as int, will be converted to tensor by parent)
            for idx, (input_ids, attention_mask, labels) in enumerate(zip(feature["all_input_ids"], feature["all_attention_mask"], feature["all_labels"])):
                passage_images = all_passage_images[idx] if idx < len(all_passage_images) else feature["images"]
                # Extract image paths (images can be list of paths or single path)
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

        # Call parent to process concatenated features
        # Note: parent's __call__ will pop("images") from features, so we need to save paths before
        batch = super().__call__(concatenated_features)
        
        # Store image paths in batch for debugging
        # To avoid AttributeError when accelerate tries to move lists to device,
        # we encode the paths as bytes and convert to tensor (inefficient but works)
        # passage_image_paths_batch is a list of lists: [[passage_0_paths, passage_1_paths, ...], ...]
        # Since passages are concatenated, we need to flatten and match the order
        if len(passage_image_paths_batch) > 0:
            # For simplicity, store first feature's paths (assuming batch_size=1 per feature in BEFT)
            if len(passage_image_paths_batch) == 1:
                all_paths = passage_image_paths_batch[0]
            else:
                # Multiple features - flatten all
                all_paths = []
                for feature_paths in passage_image_paths_batch:
                    all_paths.extend(feature_paths)
            
            # Encode paths as a string (using a separator that won't appear in paths)
            # Format: "path1|||path2|||path3" for each passage, separated by ":::"
            # Always encode all passages, even if some have empty paths
            encoded_paths = []
            for passage_paths in all_paths:
                if isinstance(passage_paths, list):
                    # Filter out None and empty strings, but keep the list structure
                    valid_paths = [str(p) for p in passage_paths if p]
                    # Join paths with ||| separator (empty string if no paths)
                    passage_str = "|||".join(valid_paths)
                else:
                    passage_str = str(passage_paths) if passage_paths else ""
                # Always append, even if empty, to maintain passage order
                encoded_paths.append(passage_str)
            
            # Join all passages with ::: separator
            # This ensures we have exactly K passages encoded
            all_paths_str = ":::".join(encoded_paths)
            
            # Convert string to bytes and then to tensor (can be moved to device safely)
            path_bytes = all_paths_str.encode('utf-8')
            batch["_passage_image_paths_tokenized"] = torch.tensor(list(path_bytes), dtype=torch.long)
        
        # Store deflection labels in batch (one per original feature, not per passage)
        # In BEFT, typically batch_size=1 per feature, so we have one deflection label for K passages
        if deflection_labels:
            batch["deflection"] = torch.tensor(deflection_labels, dtype=torch.long)
        
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

