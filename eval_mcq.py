import json
from typing import Dict, List, Union
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm

def multiple_choice(inp: Dict[str, Union[str, List[str], int]]) -> Dict[str, str]:
    PROMPT_FORMAT = '{query}\nOptions:{options}\nAnswer: '
    options = ''
    assert isinstance(inp['choices'], List)
    for option in inp['choices']:
        options += f'\n - {option}'
    query = inp['query']

    assert isinstance(inp['gold'], int)
    return {
        'prompt': PROMPT_FORMAT.format(query=query, options=options),
        'response': inp['choices'][inp['gold']],
    }

def load_model(model_path):
    model = AutoModelForCausalLM.from_pretrained(model_path)
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except:
        tokenizer = AutoTokenizer.from_pretrained("gpt2-medium")
    if tokenizer.pad_token_id is None:
        print("Setting pad token id to eos token id")
        tokenizer.pad_token = tokenizer.eos_token

    model = model.cuda()    

    return model, tokenizer

def read_jsonl(data_path):
    with open(data_path, 'r', encoding='utf-8') as file:
        return [json.loads(line) for line in file]

def get_model_predictions(model, tokenizer, formatted_questions):
    inputs = tokenizer(formatted_questions, padding=True, return_tensors='pt')
    with torch.no_grad():
        inputs = {k: v.cuda() for k, v in inputs.items()}
        outputs = model(**inputs)
    
    # get loss and normalized loss (loss divided by length of input) for each question
    shifted_logits = outputs.logits[..., :-1, :].contiguous()
    labels = inputs['input_ids'][:, 1:].contiguous()
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    losses = loss_fct(shifted_logits.view(-1, shifted_logits.size(-1)), labels.view(-1))
    normalized_losses = losses.view(-1, inputs['input_ids'].size(1) - 1).mean(-1)
    losses = losses.view(-1, inputs['input_ids'].size(1) - 1).sum(-1)
    return normalized_losses, losses



def evaluate_mcq(model_path, data_path, batch_size = 32):
    model, tokenizer = load_model(model_path)
    questions = read_jsonl(data_path)
    correct = 0; correct_normalized = 0
    pbar = tqdm(total=len(questions))
    for i in range(0, len(questions), batch_size):
        question_batch = questions[i:i+batch_size]
        formatted_questions_batch = []
        #assert number of choices is same for all questions
        for question in question_batch:
            mc_data = multiple_choice(question)
            formatted_questions = [
                mc_data['prompt'] + question['choices'][i] for i in range(len(question['choices']))
            ]
            formatted_questions_batch += formatted_questions
        normalized_losses, losses = get_model_predictions(model, tokenizer, formatted_questions_batch)
        
        num_choices_seen = 0
        for question in question_batch:
            choices_in_question = len(question['choices'])
            normalized_loss = normalized_losses[num_choices_seen:num_choices_seen+choices_in_question]
            loss = losses[num_choices_seen:num_choices_seen+choices_in_question]
            num_choices_seen += choices_in_question
            
            norm_prediction, prediction = torch.argmin(normalized_loss).item(), torch.argmin(loss).item()
            if prediction == question['gold']:
                correct += 1
            if norm_prediction == question['gold']:
                correct_normalized += 1

        # display accuracy and normalized accuracy on tqdm using pbar to 4 floats
        pbar.update(batch_size)
        pbar.set_description(
            f'accuracy: {correct / (i + batch_size):.4f}, normalized accuracy: {correct_normalized / (i + batch_size):.4f}'
        )

    accuracy = correct / len(questions)
    normalized_accuracy = correct_normalized / len(questions)
    return accuracy, normalized_accuracy


if __name__ == "__main__":
    path = "gpt2-medium"
    import sys
    dataset = sys.argv[1]
    path = f"/home/pratyus2/scratch/projects/finetune_lm/{dataset}"
    model_accuracy = evaluate_mcq(path, f"/home/pratyus2/scratch/projects/finetune_lm/datasets/{dataset}/train.jsonl")
    print(f"Model Accuracy: {model_accuracy}")

