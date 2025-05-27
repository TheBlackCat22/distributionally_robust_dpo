import os
import argparse
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from datasets import load_dataset, Dataset


def create_preferences(data, args):
    data = data.batch(batch_size=args.best_of_n)
    new_data = defaultdict(list)
    for batch in tqdm(data, desc='Creating preferences'):
        chosen_idx = np.argmax(batch['mixed Score'])
        rejected_idx = np.argmin(batch['mixed Score'])
        new_data['prompt'].append(batch[args.input_key][0])
        new_data['chosen'].append(batch[args.output_key][chosen_idx])
        new_data['chosen_score'].append(batch['mixed Score'][chosen_idx])
        new_data['rejected'].append(batch[args.output_key][rejected_idx])
        new_data['rejected_score'].append(batch['mixed Score'][rejected_idx])
    new_data = Dataset.from_dict(new_data)
    return new_data


def mix_scores(data, args):
    mixed_score = np.zeros(len(data))
    for objective in reward_objectives:
        mixed_score += np.array(data[f"{objective} Score"])*getattr(args, objective.replace('-', '_'))
    data = data.add_column(f"mixed Score", mixed_score)
    return data


def standardize_scores(data):
    score_cols = [col for col in data.column_names if " Score" in col]
    for score_col in score_cols:
        score = np.array(data[score_col])
        data = data.remove_columns(score_col)
        data = data.add_column(score_col, ((score - score.mean())/score.std()).tolist())
        print(f'{score_col} | mean={score.mean()} | std={score.std()} |')
    return data


def main(args):
    completions_data = load_dataset('json', data_dir=args.completions)
    
    for idx in range(1, len(completions_data['test'])):
        if completions_data['test'][args.input_key][idx-1]!=completions_data['test'][args.input_key][idx]:
            args.best_of_n = idx
            print(f'Detected best_of_n = {args.best_of_n}')
            break

    for split in ['test', 'train']:
        print('\n\n**************')
        print(f'{split} Split')
        print('**************')
        data = completions_data[split]
        data = standardize_scores(data)
        data = mix_scores(data, args)
        data = create_preferences(data, args)
        data.to_json(os.path.join(args.output_path, f"{split}.jsonl"))
        

if __name__ =='__main__':

    reward_objectives = ['helpsteer-helpfulness','helpsteer-correctness','helpsteer-coherence','helpsteer-complexity','helpsteer-verbosity','ultrafeedback-overall_score','ultrafeedback-instruction_following','ultrafeedback-truthfulness','ultrafeedback-honesty','ultrafeedback-helpfulness','beavertails-is_safe','prometheus-score','argilla-overall_quality','argilla-judge_lm','code-complexity','code-style','code-explanation','code-instruction-following','code-readability', 'ArmoRM']

    parser = argparse.ArgumentParser()
    parser.add_argument("--completions", type=str, default='datasets/helpsteer2_completions')
    parser.add_argument("--output_path", type=str, default='datasets/helpsteer2_preferences')
    parser.add_argument("--input_key", type=str, default='prompt')
    parser.add_argument("--output_key", type=str, default='completion')
    for objective in reward_objectives:
        parser.add_argument(f"--{objective}", type=float, default=0)
    args = parser.parse_args()
    
    weight_sum = 0
    for objective in reward_objectives:
        weight_sum+=getattr(args, objective.replace('-', '_'))
    assert weight_sum=1, f"Total Reward Objective weight must be 1. It is currently {weight_sum}"


    main(args)
    
