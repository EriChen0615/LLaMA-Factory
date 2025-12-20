# Copyright 2024 the LlamaFactory team.
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

from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple, Union

from ...extras.constants import IGNORE_INDEX
from ...extras.logging import get_logger
from .processor_utils import greedy_knapsack, infer_seqlen


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, ProcessorMixin

    from ...hparams import DataArguments
    from ..mm_plugin import ImageInput, VideoInput
    from ..template import Template

import torch
from copy import deepcopy

logger = get_logger(__name__)


def _expand_prompt_with_passages(prompt: Sequence[Dict[str, str]], passages: Sequence[Dict[str, Any]], response: Sequence[Dict[str, str]]) -> Tuple[Sequence[Dict[str, str]], Sequence[Dict[str, str]]]:
    """
    Expand prompt with passages. In BEFT, passages are dictionaries with "text" and "image" keys.
    """
    all_prompts = [{}] * len(passages)
    for psg_idx, psg in enumerate(passages):
        this_prompt = deepcopy(prompt)
        prompt_content = this_prompt[-1]['content']
        # Extract text from passage dictionary
        passage_text = psg.get("text", "") if isinstance(psg, dict) else str(psg)
        prompt_content_with_passage = prompt_content.replace("<<<EVIDENCE>>>", passage_text)
        this_prompt[-1]['content'] = prompt_content_with_passage

        all_prompts[psg_idx] = this_prompt

    return all_prompts, [response]*len(passages)

def _encode_beft_contrastive_example(
    prompt: Sequence[Dict[str, str]],
    response: Sequence[Dict[str, str]],
    passages: Sequence[Dict[str, Any]],
    system: Optional[str],
    tools: Optional[str],
    images: Sequence["ImageInput"],
    videos: Sequence["VideoInput"],
    gt_passage_idx: Union[int, List[int]],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    cutoff_len: int,
    train_on_prompt: bool,
    mask_history: bool,
) -> Tuple[List[int], List[int], List[int], List[List["ImageInput"]]]:
    """
    Encode BEFT contrastive example. Each passage has its own images list: only passage-specific images.
    No main images are used in this setup - each passage has its own distinct image.
    Returns all_input_ids, all_attention_masks, all_labels, and all_passage_images.
    """
    prompt_with_passages, exploded_responses = _expand_prompt_with_passages(prompt, passages, response)
    all_input_ids = []
    all_attention_masks = []
    all_labels = []
    all_passage_images = []

    for psg_idx, (prompt, response) in enumerate(zip(prompt_with_passages, exploded_responses)):
        # For each passage, use only passage-specific images (no main images)
        # In this setup, each passage has its own image, and there's no shared main image
        
        passage = passages[psg_idx]
        passage_images = list(images)
        
        # Get passage image if it exists (field name is "images" in the data)
        if isinstance(passage, dict) and "images" in passage and passage["images"] is not None:
            passage_image = passage["images"]
            # passage["images"] is already a list, so extend directly
            if isinstance(passage_image, (list, tuple)):
                passage_images.extend(passage_image)
            else:
                passage_images.append(passage_image)
        
        all_passage_images.append(passage_images)
        
        messages = template.mm_plugin.process_messages(prompt + response, passage_images, videos, processor)
        input_ids, labels = template.mm_plugin.process_token_ids([], [], passage_images, videos, tokenizer, processor)
        
        encoded_pairs = template.encode_multiturn(tokenizer, messages, system, tools)
        total_length = len(input_ids) + (1 if template.efficient_eos else 0)
        if mask_history:
            encoded_pairs = encoded_pairs[::-1]  # high priority for last turns

        for turn_idx, (source_ids, target_ids) in enumerate(encoded_pairs):
            if total_length >= cutoff_len:
                break

            source_len, target_len = infer_seqlen(len(source_ids), len(target_ids), cutoff_len - total_length)
            source_ids = source_ids[:source_len]
            target_ids = target_ids[:target_len]
            total_length += source_len + target_len

            if train_on_prompt:
                source_label = source_ids
            elif template.efficient_eos:
                source_label = [tokenizer.eos_token_id] + [IGNORE_INDEX] * (source_len - 1)
            else:
                source_label = [IGNORE_INDEX] * source_len

            if mask_history and turn_idx != 0:  # train on the last turn only
                target_label = [IGNORE_INDEX] * target_len
            else:
                target_label = target_ids

            if mask_history:  # reversed sequences
                input_ids = source_ids + target_ids + input_ids
                labels = source_label + target_label + labels
            else:
                input_ids += source_ids + target_ids
                labels += source_label + target_label

        if template.efficient_eos:
            input_ids += [tokenizer.eos_token_id]
            labels += [tokenizer.eos_token_id]

        all_input_ids.append(input_ids)
        all_attention_masks.append([1] * len(input_ids))
        all_labels.append(labels)

    return all_input_ids, all_attention_masks, all_labels, all_passage_images

def preprocess_beft_contrastive_dataset(
    examples: Dict[str, List[Any]],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    data_args: "DataArguments",
) -> Dict[str, List[Any]]:
    """
    Preprocess BEFT contrastive dataset. Passages are dictionaries with "text" and "images" keys.
    """
    # build inputs with format `<bos> X Y <eos>` and labels with format `<ignore> ... <ignore> Y <eos>`
    # for multiturn examples, we only mask the prompt part in each prompt-response pair.
    model_inputs = defaultdict(list)
    for i in range(len(examples["_prompt"])):
        if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) != 1:
            logger.warning("Dropped invalid example: {}".format(examples["_prompt"][i] + examples["_response"][i]))
            continue

        all_input_ids, all_attention_mask, all_labels, all_passage_images = _encode_beft_contrastive_example(
            prompt=examples["_prompt"][i],
            response=examples["_response"][i],
            passages=examples["_passages"][i],
            system=examples["_system"][i],
            tools=examples["_tools"][i],
            images=examples["_images"][i] or [],
            videos=examples["_videos"][i] or [],
            gt_passage_idx=examples["_gt_passage_idx"][i],
            template=template,
            tokenizer=tokenizer,
            processor=processor,
            cutoff_len=data_args.cutoff_len,
            train_on_prompt=data_args.train_on_prompt,
            mask_history=data_args.mask_history,
        )
        model_inputs["all_input_ids"].append(all_input_ids)
        model_inputs["all_attention_mask"].append(all_attention_mask)
        # Store passage-specific images for each passage
        model_inputs["all_passage_images"].append(all_passage_images)
        model_inputs["videos"].append(examples["_videos"][i])
        model_inputs["all_labels"].append(all_labels)
        model_inputs["gt_passage_idx"].append(examples["_gt_passage_idx"][i])

    return model_inputs

def print_beft_contrastive_dataset_example(example: Dict[str, List[int]], tokenizer: "PreTrainedTokenizer") -> None:
    print("================================================")
    print("Inspecting BEFT example:")
    print(f"\tNumber of input_ids: {len(example['all_input_ids'])}")
    print(f"\tgt_passage_idx: {example['gt_passage_idx']}")
    print(f"\tgt_input: {tokenizer.decode(example['all_input_ids'][0], skip_special_tokens=True)}")

