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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

from ...extras.constants import IGNORE_INDEX
from ...extras.logging import get_logger
from .processor_utils import DatasetProcessor, infer_seqlen

from copy import deepcopy

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, ProcessorMixin

    from ...hparams import DataArguments
    from ..mm_plugin import AudioInput, ImageInput, VideoInput
    from ..template import Template


logger = get_logger(__name__)


def _expand_prompt_with_passages_pairwise(prompt: Sequence[Dict[str, str]], passages: Sequence[Dict[str, str]], response: Sequence[Dict[str, str]]) -> Tuple[Sequence[Dict[str, str]], Sequence[Dict[str, str]]]:
    K = len(passages)
    all_prompts = [{}] * K * 2 # first half is chosen, second half is rejected
    all_responses = [response[0]] * K + [response[1]] * K
    for psg_idx, psg in enumerate(passages):
        this_prompt = deepcopy(prompt)
        prompt_content = this_prompt[-1]['content']
        prompt_content_with_passage = prompt_content.replace("<<<EVIDENCE>>>", psg)
        this_prompt[-1]['content'] = prompt_content_with_passage

        all_prompts[psg_idx] = this_prompt
        all_prompts[psg_idx + K] = this_prompt

    return all_prompts, all_responses


def _encode_bepo_pairwise_example(
    prompt: Sequence[Dict[str, str]],
    response: Sequence[Dict[str, str]],
    passages: Sequence[Dict[str, str]],
    system: Optional[str],
    tools: Optional[str],
    images: Sequence["ImageInput"],
    videos: Sequence["VideoInput"],
    audios: Sequence["AudioInput"],
    gt_passage_idx: int,
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    cutoff_len: int,
) -> Tuple[List[int], List[int], List[int], List[int]]:
    prompt_with_passages, exploded_responses = _expand_prompt_with_passages_pairwise(prompt, passages, response)
    K = len(passages)

    all_chosen_input_ids = []
    all_chosen_attention_masks = []
    all_chosen_labels = []

    all_rejected_input_ids = []
    all_rejected_attention_masks = [] 
    all_rejected_labels = []

    for i in range(K):
        prompt = prompt_with_passages[i]
        chosen_response = exploded_responses[i]
        rejected_response = exploded_responses[i + K]

        chosen_messages = template.mm_plugin.process_messages(prompt + [chosen_response], images, videos, audios, processor)
        rejected_messages = template.mm_plugin.process_messages(prompt + [rejected_response], images, videos, audios, processor)
        prompt_ids, chosen_ids = template.encode_oneturn(tokenizer, chosen_messages, system, tools)
        _, rejected_ids = template.encode_oneturn(tokenizer, rejected_messages, system, tools)
        if template.efficient_eos:
            chosen_label_ids += [tokenizer.eos_token_id]
            rejected_label_ids += [tokenizer.eos_token_id]
        
        source_len, target_len = infer_seqlen(len(prompt_ids), max(len(chosen_ids), len(rejected_ids)), cutoff_len)
        prompt_ids = prompt_ids[:source_len]
        chosen_ids = chosen_ids[:target_len]
        rejected_ids = rejected_ids[:target_len]

        chosen_input_ids = prompt_ids + chosen_ids
        chosen_labels = [IGNORE_INDEX] * source_len + chosen_ids
        rejected_input_ids = prompt_ids + rejected_ids
        rejected_labels = [IGNORE_INDEX] * source_len + rejected_ids

        all_chosen_input_ids.append(chosen_input_ids)
        all_chosen_labels.append(chosen_labels)
        all_chosen_attention_masks.append([1] * len(chosen_input_ids))
        all_rejected_input_ids.append(rejected_input_ids)
        all_rejected_labels.append(rejected_labels)
        all_rejected_attention_masks.append([1] * len(rejected_input_ids))

    return all_chosen_input_ids, all_chosen_attention_masks, all_chosen_labels, all_rejected_input_ids, all_rejected_attention_masks, all_rejected_labels


def preprocess_bepo_pairwise_dataset(
    examples: Dict[str, List[Any]],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    data_args: "DataArguments",
) -> Dict[str, List[Any]]:
    # build input pairs with format `<bos> X`, `Y1 <eos>` and `Y2 <eos>`
    model_inputs = defaultdict(list)
    for i in range(len(examples["_prompt"])):
        if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) < 2:
            logger.warning("Dropped invalid example: {}".format(examples["_prompt"][i] + examples["_response"][i]))
            continue

        all_chosen_input_ids, all_chosen_attention_masks, all_chosen_labels, all_rejected_input_ids, all_rejected_attention_masks, all_rejected_labels = _encode_bepo_pairwise_example(
            prompt=examples["_prompt"][i],
            response=examples["_response"][i],
            passages=examples["_passages"][i],
            system=examples["_system"][i],
            tools=examples["_tools"][i],
            images=examples["_images"][i] or [],
            videos=examples["_videos"][i] or [],
            audios=examples.get("_audios", [None] * len(examples["_prompt"]))[i] or [],
            gt_passage_idx=examples["_gt_passage_idx"][i] or -1,
            template=template,
            tokenizer=tokenizer,
            processor=processor,
            cutoff_len=data_args.cutoff_len,
        )
        model_inputs["chosen_input_ids"].append(all_chosen_input_ids)
        model_inputs["chosen_attention_mask"].append(all_chosen_attention_masks)
        model_inputs["chosen_labels"].append(all_chosen_labels)
        model_inputs["rejected_input_ids"].append(all_rejected_input_ids)
        model_inputs["rejected_attention_mask"].append(all_rejected_attention_masks)
        model_inputs["rejected_labels"].append(all_rejected_labels)
        model_inputs["images"].append(examples["_images"][i])
        model_inputs["videos"].append(examples["_videos"][i])
        model_inputs["gt_passage_idx"].append(examples["_gt_passage_idx"][i])

    return model_inputs


def print_bepo_pairwise_dataset_example(example: Dict[str, List[int]], tokenizer: "PreTrainedTokenizer") -> None:
    valid_chosen_labels = list(filter(lambda x: x != IGNORE_INDEX, example["chosen_labels"][0]))
    valid_rejected_labels = list(filter(lambda x: x != IGNORE_INDEX, example["rejected_labels"][0]))
    # print("chosen_input_ids:\n{}".format(example["chosen_input_ids"]))
    print("chosen_inputs:\n{}".format(tokenizer.decode(example["chosen_input_ids"][0], skip_special_tokens=False)))
    # print("chosen_label_ids:\n{}".format(example["chosen_labels"]))
    print("chosen_labels:\n{}".format(tokenizer.decode(valid_chosen_labels, skip_special_tokens=False)))
    # print("rejected_input_ids:\n{}".format(example["rejected_input_ids"]))
    print("rejected_inputs:\n{}".format(tokenizer.decode(example["rejected_input_ids"][0], skip_special_tokens=False)))
    # print("rejected_label_ids:\n{}".format(example["rejected_labels"]))
    print("rejected_labels:\n{}".format(tokenizer.decode(valid_rejected_labels, skip_special_tokens=False)))
    print("gt_passage_idx:\n{}".format(example["gt_passage_idx"]))
    print("number of passsages (K):\n{}".format(len(example["chosen_input_ids"])))


@dataclass
class BepoPairwiseDatasetProcessor(DatasetProcessor):
    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        return preprocess_bepo_pairwise_dataset(examples, self.template, self.tokenizer, self.processor, self.data_args)

    def print_data_example(self, example: dict[str, list[int]]) -> None:
        print_bepo_pairwise_dataset_example(example, self.tokenizer)
