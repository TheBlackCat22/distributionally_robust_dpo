import os
import torch
import argparse
from tqdm import tqdm
from collections import defaultdict
from torch import distributed as dist
from openrlhf.utils import get_strategy
from openrlhf.datasets import SFTDataset
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def batch_generate(args):

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    llm = LLM(
        model=args.pretrain,
        seed=args.seed,
        trust_remote_code=True,
        max_num_seqs=512,
        enable_prefix_caching=True,
        task='generate'
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrain, 
        use_fast=True
    )

    prompts_data = load_dataset('json', data_dir=args.dataset, split=args.dataset_split)
    prompts = tokenizer.apply_chat_template(prompts_data[args.input_key], add_generation_prompt=True)
    prompts = [TokensPrompt(prompt_token_ids=prompt) for prompt in prompts for _ in range(args.best_of_n)]

    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        top_p=args.top_p,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        truncate_prompt_tokens=args.prompt_max_len,
        min_tokens=1
    )
    outputs = llm.generate(prompts, sampling_params)

    completions_data = Dataset.from_dict({
        args.input_key:[prompt for prompt in prompts_data[args.input_key] for _ in range(args.best_of_n)], 
        args.output_key:[[{'content':output.outputs[0].text, 'role':'assistant'}] for output in tqdm(outputs, desc='Creating dataset')]
    })
    completions_data.to_json(os.path.join(args.output_path, f"{args.dataset_split}.jsonl"))


def batch_rm_inference(args):

    reward_objectives = ['helpsteer-helpfulness','helpsteer-correctness','helpsteer-coherence','helpsteer-complexity','helpsteer-verbosity','ultrafeedback-overall_score','ultrafeedback-instruction_following','ultrafeedback-truthfulness','ultrafeedback-honesty','ultrafeedback-helpfulness','beavertails-is_safe','prometheus-score','argilla-overall_quality','argilla-judge_lm','code-complexity','code-style','code-explanation','code-instruction-following','code-readability', 'ArmoRM']

    strategy = get_strategy(args)
    strategy.setup_distributed()

    llm = AutoModelForSequenceClassification.from_pretrained(
        'models/ArmoRM',
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if args.flash_attn else "eager",
        device_map=None
    )
    llm = strategy.prepare(llm)
    llm.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        'models/ArmoRM', 
        use_fast=True
    )

    completions_data = load_dataset('json', data_files=os.path.join(args.dataset, f'{args.dataset_split}.jsonl'))['train']
    completions = [row[args.input_key] + row[args.output_key] for row in completions_data]
    chunk_size = len(completions) // dist.get_world_size()
    start = dist.get_rank() * chunk_size
    end = start + chunk_size if dist.get_rank()!=dist.get_world_size()-1 else len(completions)
    completions = completions[start:end]
    pbar = tqdm(
        range(0, len(completions), args.micro_batch_size),
        desc='Processed Completions',
        disable=not dist.get_rank()==0
    )
    dist.barrier()
    
    outputs = [0]*len(completions)
    with torch.no_grad():
        for start_idx in pbar:
            end_idx = min(len(completions), start_idx+args.micro_batch_size)
            token_ids = tokenizer.apply_chat_template(
                completions[start_idx:end_idx], 
                return_tensors="pt",
                padding=True,
                truncation=True, 
                max_length=args.max_len
            ).to(dist.get_rank())
            output = llm(token_ids)
            output = torch.cat([output.rewards.float().cpu(), output.score.unsqueeze(-1).float().cpu()], dim=-1)
            outputs[start_idx:end_idx] = output.tolist()
    outputs = torch.tensor(outputs)
    dist.barrier()

    rewards_data = dict()
    for idx, objective in enumerate(reward_objectives):
        rewards_data[f"{objective} Score"] = outputs[:,idx].tolist()
    rewards_data = Dataset.from_dict(rewards_data)
    rewards_data.to_json(os.path.join(args.output_path, f"{args.dataset_split}_{str(dist.get_rank())}.jsonl"))
    dist.barrier()

    if dist.get_rank()==0:
        rewards_data_files = [os.path.join(args.output_path, f"{args.dataset_split}_{rank}.jsonl") for rank in range(dist.get_world_size())]
        rewards_data = concatenate_datasets([load_dataset('json', data_files=path)['train'] for path in rewards_data_files], axis=0)
        rewards_data = concatenate_datasets([completions_data, rewards_data], axis=1)
        rewards_data.to_json(os.path.join(args.output_path, f"{args.dataset_split}.jsonl"))
        for path in rewards_data_files:
            os.remove(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Common Args
    parser.add_argument("--eval_task", type=str, default=None, help="Set to generate or rm")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--dataset_split", type=str, default="test")
    parser.add_argument("--input_key", type=str, default="prompt", help="JSON dataset key")
    parser.add_argument("--output_key", type=str, default="completion", help="JSON dataset key")
    parser.add_argument("--output_path", type=str, default=None, help="Output JSON data path")

    # For generate
    parser.add_argument("--pretrain", type=str, default=None, help="HF pretrain model name or path")
    parser.add_argument("--best_of_n", type=int, default=1, help="Number of responses to generate per prompt")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Max new tokens in generation")
    parser.add_argument("--top_p", type=float, default=1.0, help="top_p for Sampling")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature for Sampling")
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--prompt_max_len", type=int, default=1024, help="Max tokens for prompt")
    
    # For rm
    parser.add_argument("--zero_stage", type=int, default=0, help="DeepSpeed ZeRO Stage")
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for deepspeed cli")
    parser.add_argument("--bf16", action="store_true", default=False, help="Enable bfloat16 for deepspeed")
    parser.add_argument("--flash_attn", action="store_true", default=False, help="Enable FlashAtten2")
    parser.add_argument("--micro_batch_size", type=int, default=64)
    parser.add_argument("--apply_chat_template", action="store_true", default=False, help="HF tokenizer apply_chat_template")
    parser.add_argument("--max_len", type=int, default=2048, help="Max tokens for the samples")

    args = parser.parse_args()
    if args.eval_task and args.eval_task == "generate":
        batch_generate(args)
    elif args.eval_task and args.eval_task == "rm":
        batch_rm_inference(args)
    else:
        print("Invalid or missing '--eval_task' argument. Please specify either 'generate' or 'rm'.")