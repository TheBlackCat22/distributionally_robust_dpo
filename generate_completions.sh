#!/bin/bash

only_test="false"
save_path="datasets/helpsteer2_completions"
for i in "$@"; do
  case "$i" in
    --only_test=*)
      only_test="${i#*=}"
      ;;
    --save_path=*)
      save_path="${i#*=}"
      ;;
    --model_path=*)
      model_path="${i#*=}"
      ;;
    --best_of_n=*)
      best_of_n="${i#*=}"
      ;;
    --temperature=*)
      temperature="${i#*=}"
      ;;
  esac
done


source $(conda info --base)/etc/profile.d/conda.sh
conda activate drdpo_env
export VLLM_CONFIGURE_LOGGING=0


run_command="python src/generate_completions.py \
--eval_task generate \
--dataset datasets/helpsteer2_prompts \
--dataset_split test \
--output_path $save_path \
--pretrain $model_path \
--best_of_n $best_of_n \
--temperature $temperature
"
echo -e "\n\n\nRun Command:"
echo $run_command
$run_command

if [ "$only_test" != "true" ]; then
  run_command="python src/generate_completions.py \
  --eval_task generate \
  --dataset datasets/helpsteer2_prompts \
  --dataset_split train \
  --output_path $save_path \
  --pretrain $model_path \
  --best_of_n $best_of_n \
  --temperature $temperature
  "
  echo -e "\n\n\nRun Command:"
  echo $run_command
  $run_command
fi


run_command="deepspeed src/generate_completions.py \
--eval_task rm \
--dataset $save_path \
--dataset_split test \
--output_path $save_path \
--bf16 \
--flash_attn \
--micro_batch_size 64 \
--apply_chat_template 
"
echo -e "\n\n\nRun Command:"
echo $run_command
$run_command

if [ "$only_test" != "true" ]; then
  run_command="deepspeed src/generate_completions.py \
  --eval_task rm \
  --dataset $save_path \
  --dataset_split train \
  --output_path $save_path \
  --bf16 \
  --flash_attn \
  --micro_batch_size 64 \
  --apply_chat_template 
  "
  echo -e "\n\n\nRun Command:"
  echo $run_command
  $run_command
fi
