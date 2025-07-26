import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification


def get_helpsteer2_prompts_dataset(split):
    if split == 'test':
        split = 'validation'
    dataset = load_dataset('nvidia/HelpSteer2', split=split)
    dataset = dataset.select(range(0, len(dataset), 2))
    dataset = dataset.remove_columns(dataset.column_names[1:])
    dataset = dataset.map(lambda example: {'prompt':[{'role':'user','content':example['prompt']}]})
    return dataset
    

def get_ultrafeedback_preferences_dataset(split):
    if split == 'train':
        split = 'train_prefs'
    elif split == 'test':
        split = 'test_prefs'
    dataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split=split)
    dataset = dataset.map(lambda example: {'prompt':[example['chosen'][0],], 'chosen':[example['chosen'][1],], 'chosen_score':example['score_chosen'], 'rejected':[example['rejected'][1],], 'rejected_score':example['score_rejected']}, remove_columns=dataset.column_names)
    return dataset
    

def main(config):

    os.makedirs('datasets', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)

    print('******************')
    print('Downloading Models')
    print('******************')

    for model_name, model_config in config['Models'].items():

        model_path = os.path.join('models', model_name)

        if not os.path.exists(model_path):
            print(f'Downloading {model_name}')

            if model_config['model_type'] == 'actor':
                tokenizer = AutoTokenizer.from_pretrained(
                    model_config['tokenizer'],
                    trust_remote_code=True
                )
                model = AutoModelForCausalLM.from_pretrained(
                    model_config['weights'],
                    trust_remote_code=True,
                    torch_dtype="auto"
                )
            elif model_config['model_type'] == 'reward':
                tokenizer = AutoTokenizer.from_pretrained(
                    model_config['tokenizer'],
                    trust_remote_code=True,
                    truncation_side="left"
                )
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_config['weights'], 
                    trust_remote_code=True,
                    torch_dtype='bfloat16'
                )
            else:
                raise ValueError("model_type should be actor or reward.")

            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id
                model.config.pad_token_id = tokenizer.pad_token_id
            model.resize_token_embeddings(len(tokenizer))

            tokenizer.save_pretrained(model_path)
            model.save_pretrained(model_path)

            print(f'{model_name} saved to {model_path}\n')
        else:
            print(f'{model_name} already exists at {model_path}\n')

    print('*********************************')
    print('Downloading & Processing Datasets')
    print('*********************************')

    for dataset_name in config['Datasets'].keys():
        
        dataset_path = os.path.join('datasets', dataset_name)
        
        if not os.path.exists(dataset_path):

            train_data = config['Datasets'][dataset_name](split='train')
            train_data.to_json(os.path.join(dataset_path, 'train.jsonl'))

            test_data = config['Datasets'][dataset_name](split='test')
            test_data.to_json(os.path.join(dataset_path, 'test.jsonl'))

            print(f'{dataset_name} saved to {dataset_path}\n')
        else:
            print(f'{dataset_name} already exists at {dataset_path}\n')


if __name__ == '__main__':
    
    config = {

        'Models' : {
            'Llama1b' : {
                'model_type' : 'actor',
                'tokenizer' : 'meta-llama/Llama-3.2-1B-Instruct',
                'weights' : 'meta-llama/Llama-3.2-1B-Instruct'
            },
            'Llama3b' : {
                'model_type' : 'actor',
                'tokenizer' : 'meta-llama/Llama-3.2-3B-Instruct',
                'weights' : 'meta-llama/Llama-3.2-3B-Instruct'
            },
            'Llama8b' : {
                'model_type' : 'actor',
                'tokenizer' : 'meta-llama/Llama-3.1-8B-Instruct',
                'weights' : 'meta-llama/Llama-3.1-8B-Instruct'
            },
            'ArmoRM' : {
                'model_type' : 'reward',
                'tokenizer' : 'RLHFlow/ArmoRM-Llama3-8B-v0.1',
                'weights' : 'RLHFlow/ArmoRM-Llama3-8B-v0.1'
            }
        },
        
        'Datasets' : {
            'helpsteer2_prompts' : get_helpsteer2_prompts_dataset,
            'ultrafeedback_preferences' : get_ultrafeedback_preferences_dataset
        },

    }

    main(config)