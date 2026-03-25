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

from .ppl_loss import build_dpp_kernel, compute_ppl_loss, compute_joint_loss, compute_ensemble_loss, dpp_subset_nll


logger = get_logger(__name__)


class DPPPriorHead(nn.Module):
    r"""Prior head for dpp_mlp modeling.

    Produces:
    - singleton_logits: passage quality logits (for DPP diagonal after sigmoid)
    - embeddings: passage similarity embeddings (for DPP off-diagonal dot products)
    """

    def __init__(self, hidden_size: int, proj_dim: int, embed_dim: int, num_layers: int) -> None:
        super().__init__()
        trunk_layers = []
        input_dim = hidden_size
        # Keep at least one linear layer in the shared trunk.
        trunk_depth = max(1, num_layers - 1)
        for _ in range(trunk_depth):
            trunk_layers.append(nn.Linear(input_dim, proj_dim))
            trunk_layers.append(nn.ReLU())
            input_dim = proj_dim
        self.trunk = nn.Sequential(*trunk_layers)
        self.singleton_head = nn.Linear(input_dim, 1)
        self.embedding_head = nn.Linear(input_dim, embed_dim)

    def forward(self, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feats = self.trunk(hidden)
        singleton_logits = self.singleton_head(feats)
        embeddings = self.embedding_head(feats)
        return singleton_logits, embeddings


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
    elif finetuning_args.ppl_prior_modeling == 'linear_head':
        input_dim = hidden_size
        proj_dim = finetuning_args.ppl_prior_head_proj_dim
        layers = []
        for _ in range(finetuning_args.ppl_prior_head_num_of_layers - 1):
            layers.append(nn.Linear(input_dim, proj_dim))
            input_dim = proj_dim
        layers.append(nn.Linear(input_dim, 1))
        prior_head = nn.Sequential(*layers)
        print(f"[PPL Trainer - Prior Head] Prior head number of layers: {finetuning_args.ppl_prior_head_num_of_layers}")
        print(f"[PPL Trainer - Prior Head] Prior head projection dimension: {proj_dim}")
        print(f"[PPL Trainer - Prior Head] Prior head parameters: {sum(p.numel() for p in prior_head.parameters())}")
    elif finetuning_args.ppl_prior_modeling == 'dpp_mlp':
        proj_dim = finetuning_args.ppl_prior_head_proj_dim
        prior_head = DPPPriorHead(
            hidden_size=hidden_size,
            proj_dim=proj_dim,
            embed_dim=finetuning_args.ppl_dpp_embed_dim,
            num_layers=finetuning_args.ppl_prior_head_num_of_layers,
        )
        print(f"[PPL Trainer - Prior Head] Prior head number of layers: {finetuning_args.ppl_prior_head_num_of_layers}")
        print(f"[PPL Trainer - Prior Head] Prior head projection dimension: {proj_dim}")
        print(f"[PPL Trainer - Prior Head] DPP embedding dimension: {finetuning_args.ppl_dpp_embed_dim}")
        print(f"[PPL Trainer - Prior Head] Prior head parameters: {sum(p.numel() for p in prior_head.parameters())}")
    else:
        print(f"[PPL Trainer - Prior Head] No Prior head")
    return prior_head

def initialize_deflection_head(finetuning_args: "FinetuningArguments", hidden_size: int):
    print(f"[PPL Trainer] Deflection head modeling: {finetuning_args.ppl_deflection_modeling}")
    print(f"[PPL Trainer] Use deflection head loss: {finetuning_args.use_deflection_head_loss}")
    print(f"[PPL Trainer] Deflection head loss factor: {finetuning_args.ppl_deflection_loss_factor}")

    deflection_head = None
    if finetuning_args.ppl_deflection_modeling in ['mlp_head']:
        # Initialize a 2-layer MLP head of shape [h]
        input_dim = hidden_size
        proj_dim = finetuning_args.ppl_deflection_head_proj_dim

        mlp_layers = []
        for i in range(finetuning_args.ppl_deflection_head_num_of_layers - 1):
            mlp_layers.append(nn.Linear(input_dim, proj_dim))
            mlp_layers.append(nn.ReLU())
            input_dim = proj_dim
        mlp_layers.append(nn.Linear(input_dim, 1))

        deflection_head = nn.Sequential(*mlp_layers)
        print(f"[PPL Trainer - Deflection Head] Deflection head number of layers: {finetuning_args.ppl_deflection_head_num_of_layers}")
        print(f"[PPL Trainer - Deflection Head] Deflection head projection dimension: {proj_dim}")
        print(f"[PPL Trainer - Deflection Head] Deflection head parameters: {sum(p.numel() for p in deflection_head.parameters())}")
        if finetuning_args.ppl_deflection_head_path is not None:
            deflection_head.load_state_dict(torch.load(finetuning_args.ppl_deflection_head_path))
            print(f"[PPL Trainer - Deflection Head] Deflection head loaded from {finetuning_args.ppl_deflection_head_path}")
        else:
            print(f"[PPL Trainer - Deflection Head] No deflection head path provided, initializing a new deflection head")
    elif finetuning_args.ppl_deflection_modeling == 'linear_head':
        input_dim = hidden_size
        proj_dim = finetuning_args.ppl_deflection_head_proj_dim
        layers = []
        for _ in range(finetuning_args.ppl_deflection_head_num_of_layers - 1):
            layers.append(nn.Linear(input_dim, proj_dim))
            input_dim = proj_dim
        layers.append(nn.Linear(input_dim, 1))
        deflection_head = nn.Sequential(*layers)
        print(f"[PPL Trainer - Deflection Head] Deflection head number of layers: {finetuning_args.ppl_deflection_head_num_of_layers}")
        print(f"[PPL Trainer - Deflection Head] Deflection head projection dimension: {proj_dim}")
        print(f"[PPL Trainer - Deflection Head] Deflection head parameters: {sum(p.numel() for p in deflection_head.parameters())}")
    else:
        print(f"[PPL Trainer - Deflection Head] No Deflection head")
    return deflection_head

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

def get_last_hidden_state_at_eos(hidden_states: "torch.Tensor", input_ids: "torch.Tensor", tokenizer) -> "torch.Tensor":
    """
    Extract hidden states at the last token position for each sequence.
    Since left padding is enforced, the last token (-1 position) is the actual last token of the sequence.
    Args:
        hidden_states: [batch_size, seq_len, hidden_size]
        input_ids: [batch_size, seq_len]
        tokenizer: tokenizer to verify left padding is enforced
    Returns:
        hidden_at_eos: [batch_size, hidden_size]
    """
    # Verify left padding is enforced
    assert tokenizer.padding_side == "left", "This method requires left padding. Set tokenizer.padding_side = 'left'"
    
    # With left padding, the last token position (-1) is the actual last token (EOS)
    # Simply extract hidden states at the last position for all sequences
    return hidden_states[:, -1, :]


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

        self.prior_head = initialize_prior_head(finetuning_args, hidden_size=self.model.config.hidden_size)
        if self.prior_head is not None:
            # Prepare prior_head with accelerator for proper distributed training and mixed precision
            self.prior_head = self.accelerator.prepare_model(self.prior_head, evaluation_mode=False)
        
        # Initialize deflection head
        self.deflection_head = initialize_deflection_head(finetuning_args, hidden_size=self.model.config.hidden_size)
        if self.deflection_head is not None:
            # Prepare deflection_head with accelerator for proper distributed training and mixed precision
            self.deflection_head = self.accelerator.prepare_model(self.deflection_head, evaluation_mode=False)

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
        
        if self.prior_head is not None and self.finetuning_args.freeze_prior_head_weights:
            print(f"[PPL Trainer] Freezing prior head weights...")
            for param in self.prior_head.parameters():
                param.requires_grad = False
        else:
            print(f"[PPL Trainer] Prior head weights are trainable")
        
        if self.deflection_head is not None:
            print(f"[PPL Trainer] Deflection head weights are trainable")
        
        print(f"[PPL Trainer] ppl_enable_chunked_checkpoint: {self.finetuning_args.ppl_enable_chunked_checkpoint}")
        print(f"[PPL Trainer] ppl_forward_chunk_size: {self.finetuning_args.ppl_forward_chunk_size}")
        
        # After training starts, check if prior_head params are in optimizer
        # param_ids_in_optimizer = {id(p) for group in self.optimizer.param_groups for p in group['params']}
        # param_ids_in_prior_head = {id(p) for p in self.prior_head.parameters()}
        # print(f"[PPL Trainer] Prior head params in optimizer: {param_ids_in_prior_head.issubset(param_ids_in_optimizer)}")
        # breakpoint()
        
        # if finetuning_args.use_ppl_loss:
            # print("Using PPL training with Posterior Loss.")
        # else:
            # print("Using PPL Trainer, but Posterior Loss is disabled")
            # print(f"Using PPL temperature (tau): {self.finetuning_args.ppl_temperature}")
            # print(f"Using PPL prior: {self.finetuning_args.ppl_prior}")
            # print(f"Using PPL llk factor: {self.finetuning_args.ppl_llk_factor}")
    
    def concatenated_forward(
        self,
        model,
        batch,
        hidden_state_offset=0,
        return_hidden_states=True,
        return_per_token_logps=False,
    ):
        r"""
        The first instance of the batch is the GT passage.
        NOTE: only batch size = 1 for the collator is supported currently.
        """
        outputs = model(**batch, return_dict=True, use_cache=False, output_hidden_states=return_hidden_states)
        all_logits = outputs["logits"]
        labels = batch["labels"]
        lengths = (labels != IGNORE_INDEX).sum(-1)

        per_token_logps = None
        if return_per_token_logps:
            labels_shifted = labels[:, 1:].clone()
            logits_shifted = all_logits[:, :-1, :]
            loss_mask = labels_shifted != IGNORE_INDEX
            labels_shifted[labels_shifted == IGNORE_INDEX] = 0  # dummy token
            per_token_logps = torch.gather(
                logits_shifted.log_softmax(-1), dim=2, index=labels_shifted.unsqueeze(2)
            ).squeeze(2)
            per_token_logps = per_token_logps * loss_mask
            pos_logps, neg_logps = per_token_logps[0].sum(-1), per_token_logps[1:].sum(-1)
        else:
            all_logps, _ = get_batch_logps(logits=all_logits, labels=labels)
            pos_logps, neg_logps = all_logps[:1], all_logps[1:]
        
        # Extract last-layer hidden states at position just before the first label
        hidden_at_pre_label = None
        hidden_at_eos = None
        if return_hidden_states:
            hidden_states = outputs["hidden_states"]
            last_hidden_states = hidden_states[-1]  # Shape: [batch_size, seq_len, hidden_size]
            hidden_at_pre_label = get_last_hidden_state_before_label(last_hidden_states, labels, hidden_state_offset=hidden_state_offset)
            # Extract hidden states at EOS token for deflection head
            hidden_at_eos = get_last_hidden_state_at_eos(last_hidden_states, batch["input_ids"], self.tokenizer)
        
        if return_per_token_logps:
            all_logits = None

        return pos_logps, neg_logps, per_token_logps, all_logits, outputs, lengths[0], hidden_at_pre_label, hidden_at_eos

    def concatenated_forward_chunk_checkpointing(self, model, batch, hidden_state_offset=0, return_hidden_states=True):
        r"""
        Chunked version of concatenated_forward with gradient checkpointing.
        Processes K passages in chunks to reduce memory while maintaining exact gradients.
        
        The first instance of the batch is the GT passage.
        NOTE: only batch size = 1 for the collator is supported currently.
        """
        from torch.utils.checkpoint import checkpoint
        
        K = batch["input_ids"].size(0)
        chunk_size = self.finetuning_args.ppl_forward_chunk_size
        
        # logger.info(f"[PPL Trainer] Using chunked forward with checkpointing: K={K}, chunk_size={chunk_size}")
        
        # Process K passages in chunks
        all_logits_list = []
        all_hidden_states_list = []
        all_hidden_at_eos_list = []
        all_per_token_logps_list = []
        last_outputs = None
        
        for start_idx in range(0, K, chunk_size):
            end_idx = min(start_idx + chunk_size, K)
            
            # Slice inputs for this chunk
            chunk_batch = {
                "input_ids": batch["input_ids"][start_idx:end_idx],
                "attention_mask": batch["attention_mask"][start_idx:end_idx],
                "labels": batch["labels"][start_idx:end_idx],
                "image_grid_thw": batch["image_grid_thw"][start_idx:end_idx],
            }
            
            # Handle pixel_values slicing (concatenated patches)
            if "pixel_values" in batch:
                # Calculate patch boundaries using image_grid_thw
                image_grid_thw = batch["image_grid_thw"]
                patches_per_passage = (image_grid_thw[:, 0] * image_grid_thw[:, 1] * image_grid_thw[:, 2]).long()
                cumsum_patches = torch.cumsum(patches_per_passage, dim=0)
                
                patch_start = 0 if start_idx == 0 else cumsum_patches[start_idx - 1].item()
                patch_end = cumsum_patches[end_idx - 1].item()
                
                chunk_batch["pixel_values"] = batch["pixel_values"][patch_start:patch_end]
            
            # Define forward function for checkpointing
            def forward_chunk(chunk_batch_dict):
                """Forward function to be checkpointed. Must be deterministic."""
                return model(
                    **chunk_batch_dict,
                    return_dict=True,
                    use_cache=False,
                    output_hidden_states=return_hidden_states
                )
            
            # Use checkpoint: activations are recomputed during backward
            outputs_chunk = checkpoint(
                forward_chunk,
                chunk_batch,
                use_reentrant=False
            )
            
            # Collect logits
            # all_logits_list.append(outputs_chunk.logits)
            last_outputs = outputs_chunk  # Keep last for return
            
            # Collect hidden states if needed
            if return_hidden_states and outputs_chunk.hidden_states is not None:
                last_hidden_states_chunk = outputs_chunk.hidden_states[-1]
                hidden_at_pre_label_chunk = get_last_hidden_state_before_label(
                    last_hidden_states_chunk,
                    chunk_batch["labels"],
                    hidden_state_offset=hidden_state_offset
                )
                all_hidden_states_list.append(hidden_at_pre_label_chunk)
                
                # Extract hidden states at EOS token for deflection head
                hidden_at_eos_chunk = get_last_hidden_state_at_eos(
                    last_hidden_states_chunk,
                    chunk_batch["input_ids"],
                    self.tokenizer
                )
                all_hidden_at_eos_list.append(hidden_at_eos_chunk)
            
            labels_shifted = chunk_batch["labels"][:, 1:].clone()
            logits_shifted = outputs_chunk.logits[:, :-1, :]
            loss_mask = labels_shifted != IGNORE_INDEX
            labels_shifted[labels_shifted == IGNORE_INDEX] = 0  # dummy token
            per_token_logps_chunk = torch.gather(logits_shifted.log_softmax(-1), dim=2, index=labels_shifted.unsqueeze(2)).squeeze(2)
            per_token_logps_chunk = per_token_logps_chunk * loss_mask  # Zero out ignored positions
            all_per_token_logps_list.append(per_token_logps_chunk)
        
        # Concatenate all logits
        # all_logits = torch.cat(all_logits_list, dim=0)  # Shape: [K, seq_len, vocab_size]
        all_per_token_logps = torch.cat(all_per_token_logps_list, dim=0)  # Shape: [K, seq_len, vocab_size]
        
        # Compute log probabilities from concatenated logits
        labels = batch["labels"]
        pos_logps, neg_logps = all_per_token_logps[0].sum(-1), all_per_token_logps[1:].sum(-1)
        lengths = (labels != IGNORE_INDEX).sum(-1)
        # all_logps, lengths = get_batch_logps(logits=all_logits, labels=labels)
        # pos_logps, neg_logps = all_logps[:1], all_logps[1:]
        
        # Concatenate hidden states
        hidden_at_pre_label = None
        hidden_at_eos = None
        if return_hidden_states and all_hidden_states_list:
            hidden_at_pre_label = torch.cat(all_hidden_states_list, dim=0)  # Shape: [K, hidden_size]
            # Concatenate EOS hidden states
            if all_hidden_at_eos_list:
                hidden_at_eos = torch.cat(all_hidden_at_eos_list, dim=0)  # Shape: [K, hidden_size]
        
        return pos_logps, neg_logps, all_per_token_logps, None, last_outputs, lengths[0], hidden_at_pre_label, hidden_at_eos

    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False, eval_mode=False):
        """
        Override `compute_loss` in `transformers.trainer`.
        """

        prior_inputs = None
        bs = inputs["input_ids"].size(0)
        K = bs
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
            K = bs//2

        # Use chunked forward with checkpointing if K exceeds chunk size
        return_hidden_states = self.finetuning_args.ppl_prior_modeling in ['mlp_head', 'linear_head', 'dpp_mlp'] or self.finetuning_args.ppl_deflection_modeling in ['mlp_head', 'linear_head']
        if self.finetuning_args.ppl_enable_chunked_checkpoint and self.finetuning_args.ppl_forward_chunk_size is not None and K > self.finetuning_args.ppl_forward_chunk_size:
            pos_logps, neg_logps, per_token_logps, all_logits, outputs, ans_len, hidden_at_pre_label, hidden_at_eos = self.concatenated_forward_chunk_checkpointing(model, inputs, self.finetuning_args.ppl_hidden_state_offset, return_hidden_states=return_hidden_states)
        else:
            pos_logps, neg_logps, _, all_logits, outputs, ans_len, hidden_at_pre_label, hidden_at_eos = self.concatenated_forward(model, inputs, self.finetuning_args.ppl_hidden_state_offset, return_hidden_states=return_hidden_states)
            per_token_logps = None

        prior_logits = None
        dpp_embeddings = None
        if self.prior_head is not None:
            if self.finetuning_args.ppl_prior_modeling in ['mlp_head', 'linear_head']:
                prior_logits = self.prior_head(hidden_at_pre_label)  # Shape: [batch_size, 1]
            elif self.finetuning_args.ppl_prior_modeling == 'dpp_mlp':
                prior_logits, dpp_embeddings = self.prior_head(hidden_at_pre_label)
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
            beft_loss, posterior_loss, llk_loss, posterior_logprob, prior_logprob = compute_ensemble_loss(all_logits, inputs["labels"], prior_logits, per_token_logps=per_token_logps)
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
            if self.finetuning_args.ppl_prior_modeling == 'dpp_mlp':
                if prior_logits is None or dpp_embeddings is None:
                    prior_loss = torch.tensor(0.0, device=self.model.device)
                else:
                    dpp_kernel = build_dpp_kernel(
                        prior_logits,
                        dpp_embeddings,
                        jitter=self.finetuning_args.ppl_dpp_jitter,
                    )
                    gt_indices = torch.tensor([0], device=dpp_kernel.device, dtype=torch.long)
                    prior_loss = dpp_subset_nll(
                        dpp_kernel,
                        gt_indices,
                        jitter=self.finetuning_args.ppl_dpp_jitter,
                    ) * prior_lambda
            elif self.finetuning_args.ppl_prior_loss_type == 'softmax':
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
        if self.optimizer is None and not self.finetuning_args.freeze_vlm_weights:
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
                    logger.info(f"Adding {len(prior_head_params)} prior_head parameters to optimizer")
                    self.optimizer.add_param_group({
                        'params': prior_head_params,
                        'lr': self.finetuning_args.prior_head_lr,
                        'weight_decay': self.args.weight_decay,
                    })
                else:
                    logger.info("Prior_head parameters already in optimizer")
        
        # Add deflection_head parameters to the optimizer if it exists and is trainable
        if self.deflection_head is not None and self.optimizer is not None:
            deflection_head_params = list(self.deflection_head.parameters())
            if deflection_head_params and deflection_head_params[0].requires_grad:
                # Check if deflection_head params are already in optimizer
                optimizer_params = set()
                for group in self.optimizer.param_groups:
                    optimizer_params.update(id(p) for p in group['params'])
                
                deflection_head_param_ids = set(id(p) for p in deflection_head_params)
                
                if not deflection_head_param_ids.issubset(optimizer_params):
                    # Add deflection_head parameters as a new param group
                    # Use same learning rate as prior head
                    logger.info(f"Adding {len(deflection_head_params)} deflection_head parameters to optimizer. LR={self.finetuning_args.prior_head_lr}")
                    self.optimizer.add_param_group({
                        'params': deflection_head_params,
                        'lr': self.finetuning_args.prior_head_lr,
                        'weight_decay': self.args.weight_decay,
                    })
                else:
                    logger.info("Deflection_head parameters already in optimizer")
        
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

    def log(self, logs: Dict[str, float], *args, **kwargs) -> None:
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
        super().log(logs, *args, **kwargs)
        
        # Clear metrics for next cycle
        self._metrics.clear()

    @override
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """
        Override save_model to save prior_head and deflection_head (mlp_head) as separate .pt files.
        """
        # Call parent save_model first
        super().save_model(output_dir, _internal_call)
        
        if output_dir is None:
            output_dir = self.args.output_dir
        
        # Ensure output directory exists (important for distributed training)
        os.makedirs(output_dir, exist_ok=True)
        
        # Save prior_head separately if it exists (only on main process)
        if self.accelerator.is_main_process and hasattr(self, 'prior_head') and self.prior_head is not None:
            prior_head_path = os.path.join(output_dir, "prior_head.pt")
            # Unwrap the model if it's wrapped by accelerator (e.g., DDP)
            prior_head_to_save = self.accelerator.unwrap_model(self.prior_head)
            torch.save(prior_head_to_save.state_dict(), prior_head_path)
            logger.info(f"Saved prior_head to {prior_head_path}")
        
        # Save deflection_head separately if it exists (only on main process)
        if self.accelerator.is_main_process and hasattr(self, 'deflection_head') and self.deflection_head is not None:
            deflection_head_path = os.path.join(output_dir, "deflection_head.pt")
            # Unwrap the model if it's wrapped by accelerator (e.g., DDP)
            deflection_head_to_save = self.accelerator.unwrap_model(self.deflection_head)
            torch.save(deflection_head_to_save.state_dict(), deflection_head_path)
            logger.info(f"Saved deflection_head to {deflection_head_path}")
