#!/bin/bash
export FORCE_TORCHRUN=1
llamafactory-cli train my_configs/norag_answer_qwen2vl-2B_full_sft.yaml