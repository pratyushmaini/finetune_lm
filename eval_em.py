import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from composer import *
from params import parse_args
from eval_mcq import load_model, read_jsonl
from tqdm import tqdm

def normalize_answer(answer):
    """ Normalize the answer for comparison. """
    try:
      gold_answer = re.search("#### (\\-?[0-9\\.\\,]+)", answer).group(1)
    except:
      gold_answer = "0"
    return gold_answer

def exact_match(gold_answer, model_answer):
    """ Check if the model answer matches the gold answer based on the specified criteria. """
    normalized_gold = normalize_answer(gold_answer)
    normalized_model = normalize_answer(model_answer)
    return normalized_gold == normalized_model

def rouge_l(gold_answer, model_answer):
    #get normalized gold answer
    normalized_gold = normalize_answer(gold_answer)
    # check if gold answer is in model answer 
    # acceptable conditions: should not have any other number or word before or after the gold answer. special characters are allowed

    #use regex
    pattern = re.compile(f"\\b{normalized_gold}\\b")
    match = re.search(pattern, model_answer)
    if match:
      return True
    else:
      return False


    

def generate_model_answer(question, model, tokenizer):
    """ Generate an answer using the model. """
    generation_kwargs = {
        'do_sample': False,
        'temperature': 0.0,
        'max_new_tokens': 200,
    }
    #question is a list of strings
    tokenizer.padding_side = "left"
    model.config.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id
    input = tokenizer(question, return_tensors='pt', padding=True, truncation=True)
    #send id, attention mask to cuda
    input_ids = input['input_ids'].cuda()
    attention_mask = input['attention_mask'].cuda()

    outputs = model.generate(input_ids=input_ids, **generation_kwargs)
    texts  = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return texts


def evaluate_em(model_path, data_path, data_composer, batch_size = 32):
    model, tokenizer = load_model(model_path)
    questions = read_jsonl(data_path)
    correct = 0; correct_recall = 0
    pbar = tqdm(total=len(questions))
    for i in range(0, len(questions), batch_size):
        question_batch = questions[i:i+batch_size]
        formatted_questions_batch = []
        num_question_tokens_batch = []
        #assert number of choices is same for all questions
        for question in question_batch:
            mc_data = data_composer(question)
            formatted_questions = [mc_data['prompt']]
            num_question_tokens = (len(tokenizer.tokenize(mc_data['prompt'])) - 1)
            num_question_tokens_batch.append(num_question_tokens)
            formatted_questions_batch += formatted_questions
        
        #send to model
        model_answers  = generate_model_answer(formatted_questions_batch, model, tokenizer)
        for answer, question in zip(model_answers, question_batch):
            answer = answer.replace(question["query"], "")
            if exact_match(question['answer'], answer):
                correct += 1
            if rouge_l(question['answer'], answer):
                correct_recall += 1
              
        # display accuracy and normalized accuracy on tqdm using pbar to 4 floats
        pbar.update(batch_size)
        n_ex = i + len(question_batch)
        pbar.set_description(
            f'accuracy: {correct / n_ex:.4f} recall: {correct_recall / n_ex:.4f}'
        )

    accuracy = correct / len(questions)
    return accuracy



if __name__ == "__main__":
    args = parse_args().parse_args()
    dataset = args.dataset
    path = args.model_path
    data_composer  = composer_dict[dataset]

    if args.do_train_eval:
        model_accuracy = evaluate_em(path, f"datasets/{dataset}/train.jsonl", data_composer, batch_size=args.total_batch_size)
        print(f"Model Accuracy on Train: {model_accuracy}")
    
    model_accuracy = evaluate_em(path, f"datasets/{dataset}/validation.jsonl", data_composer, batch_size=args.total_batch_size)
    print(f"Model Accuracy on Val: {model_accuracy}")

