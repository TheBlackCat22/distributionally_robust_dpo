#!/bin/bash

for i in "$@"; do
  case "$i" in
    --model_path=*)
      model_path="${i#*=}"
      ;;
  esac
done


source $(conda info --base)/etc/profile.d/conda.sh
conda activate drdpo_env
GPUS_ON_NODE=4

run_command="accelerate launch --multi_gpu \
--num_machines 1 \
--num_processes $GPUS_ON_NODE \
--mixed_precision bf16 \
--dynamo_backend no \
-m lm_eval \
--model hf \
--tasks leaderboard \
--model_args pretrained=$model_path,truncation=True,max_length=2048,dtype=bfloat16,trust_remote_code=True,attn_implementation=flash_attention_2,parallelize=True \
--batch_size auto:8 \
--output_path results \
--apply_chat_template \
--fewshot_as_multiturn \
--gen_kwargs max_gen_toks=1024 \
--seed 42
"
echo -e "\n\n\nRun Command:"
echo $run_command
$run_command
