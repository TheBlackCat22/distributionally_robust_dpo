#!/bin/bash

kldpo_tau=""
wdpo_rho=""
drdpo_beta=""
for i in "$@"; do
  case "$i" in
    --model_path=*)
      model_path="${i#*=}"
      ;;
    --dataset_path=*)
      dataset_path="${i#*=}"
      ;;
    --save_path=*)
      save_path="${i#*=}"
      ;;
    --kldpo_tau=*)
      kldpo_tau="--kldpo_tau ${i#*=}"
      ;;
    --wdpo_rho=*)
      wdpo_rho="--wdpo_rho ${i#*=}"
      ;;
  esac
done

if ! [ -z "$kldpo_tau" ]; then
  train_task="kldpo"
elif ! [ -z "$wdpo_rho" ]; then
  train_task="wdpo"
elif ! [ -z "$drdpo_beta" ]; then
  train_task="drdpo"
else
  train_task="dpo"
fi

if [ $train_task == "wdpo" ]; then
  micro_train_batch_size="1"
else
  micro_train_batch_size="4"
fi


source $(conda info --base)/etc/profile.d/conda.sh
conda activate drdpo_env


run_command="deepspeed src/train_preference.py \
--save_path $save_path \
--save_steps 158 \
--save_hf_ckpt \
--disable_ds_ckpt \
--eval_steps 79 \
--ckpt_path $save_path \
--micro_train_batch_size $micro_train_batch_size \
--train_batch_size 128 \
--gradient_checkpointing \
--zero_stage 2 \
--bf16 \
--learning_rate 5.0e-7 \
--lr_warmup_ratio 0.1 \
--zpg $SLURM_GPUS_ON_NODE \
--flash_attn \
--train_task $train_task \
--max_epochs 8 \
--beta 0.01 \
--pretrain $model_path \
--dataset json@$dataset_path \
--prompt_key prompt \
--apply_chat_template \
--max_len 2048 \
--use_tensorboard $save_path \
$kldpo_tau $wdpo_rho $drdpo_beta
"
echo -e "\n\n\nRun Command:"
echo $run_command
$run_command
