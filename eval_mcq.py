import json
from composer import *
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm
from params import parse_args


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

def get_model_predictions(model, tokenizer, formatted_questions, num_question_tokens):
    inputs = tokenizer(formatted_questions, padding=True, return_tensors='pt')
    with torch.no_grad():
        inputs = {k: v.cuda() for k, v in inputs.items()}
        outputs = model(**inputs)
    
    # get loss and normalized loss (loss divided by length of input) for each question
    shifted_logits = outputs.logits[..., :-1, :].contiguous()
    labels = inputs['input_ids'][:, 1:].contiguous()
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)

    #set the labels to -100 where the input is padding or is a part of the question
    labels[labels == tokenizer.pad_token_id] = -100
    for i in range(len(num_question_tokens)):
        labels[i, :num_question_tokens[i]] = -100
    
    # import ipdb; ipdb.set_trace()
    losses = loss_fct(shifted_logits.view(-1, shifted_logits.size(-1)), labels.view(-1))
    losses = losses.view(-1, inputs['input_ids'].size(1) - 1).sum(-1)
    #divide by number of non -100 labels
    normalized_losses = losses / (labels != -100).sum(-1).float()
    return normalized_losses, losses



def evaluate_mcq(model_path, data_path, data_composer, batch_size = 32):
    model, tokenizer = load_model(model_path)
    questions = read_jsonl(data_path)
    correct = 0; correct_normalized = 0
    pbar = tqdm(total=len(questions))
    for i in range(0, len(questions), batch_size):
        question_batch = questions[i:i+batch_size]
        formatted_questions_batch = []
        num_question_tokens_batch = []
        #assert number of choices is same for all questions
        for question in question_batch:
            mc_data = data_composer(question)
            formatted_questions = [
                mc_data['prompt'] + " " + question['choices'][i] for i in range(len(question['choices']))
            ]
            num_question_tokens = (len(tokenizer.tokenize(mc_data['prompt'])) - 1)
            num_question_tokens_batch.extend([num_question_tokens]*len(question['choices']))
            formatted_questions_batch += formatted_questions
        normalized_losses, losses = get_model_predictions(model, tokenizer, formatted_questions_batch, num_question_tokens_batch)
        
        num_choices_seen = 0
        
        for question in question_batch:
            # prompt = multiple_choice(question)['prompt']
            # input_ids = tokenizer(prompt, return_tensors='pt').input_ids.cuda()
            # out = model.generate(input_ids, do_sample = False, max_new_tokens = 200)
            # text_out = tokenizer.batch_decode(out)
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
        n_ex = i + len(question_batch)
        pbar.set_description(
            f'accuracy: {correct / n_ex:.4f}, normalized accuracy: {correct_normalized / n_ex:.4f}'
        )

    accuracy = correct / len(questions)
    normalized_accuracy = correct_normalized / len(questions)
    return accuracy, normalized_accuracy


if __name__ == "__main__":
    args = parse_args().parse_args()
    dataset = args.dataset
    path = args.model_path
    data_composer  = composer_dict[dataset]

    if args.do_train_eval:
        model_accuracy = evaluate_mcq(path, f"datasets/{dataset}/train.jsonl", data_composer, batch_size=args.total_batch_size)
        print(f"Model Accuracy on Train: {model_accuracy}")
    
    model_accuracy = evaluate_mcq(path, f"datasets/{dataset}/validation.jsonl", data_composer, batch_size=args.total_batch_size)
    print(f"Model Accuracy on Val: {model_accuracy}")

