from datasets import load_dataset
import json, os
from tqdm import tqdm
# each dataset should be re-written as a jsonl file with the following format:
# In case of MCQ the format should be as follows:
# {"choices": ["Sunlight is the source of energy for nearly all ecosystems.", "Most ecosystems are found on land instead of in water.", "Carbon dioxide is more available than other gases.", "The producers in all ecosystems are plants."], "query": "Question: Which statement best explains why photosynthesis is the foundation of most food webs?\n", "gold": 0}

# In case of QA the format should be as follows:
# {"query": "Question: Which statement best explains why photosynthesis is the foundation of most food webs?\n", "gold": "Sunlight is the source of energy for nearly all ecosystems."}


def commonsense_qa_preprocess():
    # {'id': '61fe6e879ff18686d7552425a36344c8', 'question': 'Sammy wanted to go to where the people were.  Where might he go?', 'question_concept': 'people', 'choices': {'label': ['A', 'B', 'C', 'D', 'E'], 'text': ['race track', 'populated areas', 'the desert', 'apartment', 'roadblock']}, 'answerKey': 'B'}
    dataset = load_dataset("commonsense_qa", streaming=True)
    jsonl_dict = {}
    for split in ["train", "validation"]:
        jsonl = []
        current_dataset = dataset[split]
        for row in current_dataset:
            question = row["question"]
            choices = row["choices"]["text"]
            answerKey = row["answerKey"]
            #convert answerKey to index based on choices["label"]
            answerKey = row["choices"]["label"].index(answerKey)
            jsonl_record = {
                "choices": choices,
                "query": f"{question}",
                "gold": answerKey
            }
            jsonl.append(jsonl_record)
        jsonl_dict[split] = jsonl
    return jsonl_dict


def piqa_preprocess():
    # {'goal': 'to be a good student', 'sol1': 'study hard', 'sol2': 'play video games', 'label': 0}
    dataset = load_dataset("piqa", streaming=True)
    jsonl_dict = {}
    for split in ["train", "validation"]:
        jsonl = []
        current_dataset = dataset[split]
        for row in current_dataset:
            question = row["goal"]
            choices = [row["sol1"], row["sol2"]]
            answerKey = row["label"]
            jsonl_record = {
                "choices": choices,
                "query": f"{question}",
                "gold": answerKey
            }
            jsonl.append(jsonl_record)
        jsonl_dict[split] = jsonl
    return jsonl_dict


def arc_easy_preprocess():
    # we will take a max of 10k examples from the train split
    # "question": Which factor will most likely cause a person to develop a fever?
    # "choices": { "text": [ "a leg muscle relaxing after exercise", "a bacterial population in the bloodstream", "several viral particles on the skin", "carbohydrates being digested in the stomach" ], "label": [ "A", "B", "C", "D" ] }
    # "answerKey": "B
    jsonl_dict = {}
    dataset = load_dataset("ai2_arc", "ARC-Easy", streaming=True)
    for split in ["train", "validation", "test"]:
        jsonl = []
        current_dataset = dataset[split]
        for row in tqdm(current_dataset):
            question = row["question"]
            choices = row["choices"]["text"]
            answerKey = row["answerKey"]
            #convert answerKey to index based on choices["label"]
            answerKey = row["choices"]["label"].index(answerKey)
            jsonl_record = {
                "choices": choices,
                "query": f"{question}",
                "gold": answerKey
            }
            jsonl.append(jsonl_record)
        jsonl_dict[split] = jsonl
    
    return jsonl_dict


def trivia_qa_preprocess():
    dataset = load_dataset("trivia_qa", "rc.nocontext", streaming=True)
    jsonl_dict = {}
    for split in ["train", "validation"]:
        jsonl = []
        current_dataset = dataset[split]
        num_samples = 0
        for row in tqdm(current_dataset):
            question = row["question"]
            choices = row["answer"]["aliases"]
            answerKey_str = row["answer"]["value"]
            if answerKey_str not in choices:
                print(f"answer {answerKey_str} not in choices")
                continue
            answerKey = choices.index(answerKey_str)
            jsonl_record = {
                "choices": choices,
                "query": f"{question}",
                "gold": answerKey
            }
            jsonl.append(jsonl_record)
            num_samples += 1
            if num_samples == 20000:
                break

        jsonl_dict[split] = jsonl
    return jsonl_dict

def gsm8k_preprocess():
    dataset = load_dataset("gsm8k", "main", streaming=True)
    jsonl_dict = {}
    for split in ["train", "test"]:
        jsonl = []
        current_dataset = dataset[split]
        for row in tqdm(current_dataset):
            question = row["question"]
            answerKey = row["answer"]
            jsonl_record = {
                "query": f"{question}",
                "answer": answerKey
            }
            jsonl.append(jsonl_record)
        jsonl_dict[split] = jsonl
    return jsonl_dict

preprocess_func = {
    # mcq datasets
    "commonsense_qa": commonsense_qa_preprocess,   
    "piqa": piqa_preprocess,
    "arc_easy": arc_easy_preprocess,
    "trivia_qa": trivia_qa_preprocess,
    "gsm8k": gsm8k_preprocess,
}

def get_dataset(dataset_name):
    # now preprocess the dataset
    jsonl_dict = preprocess_func[dataset_name]()
    #save the jsonl files 
    os.makedirs(f"datasets/{dataset_name}", exist_ok=True)
    for split, jsonl in jsonl_dict.items():
        with open(f"datasets/{dataset_name}/{split}.jsonl", "w", encoding='utf-8') as f:
            data = jsonl
            for entry in data:
                json_record = json.dumps(entry, ensure_ascii=False)
                f.write(json_record + '\n')
    return jsonl 

if __name__ == "__main__":
    get_dataset("gsm8k")
    # get_dataset("arc_easy")
    # get_dataset("trivia_qa")
    # get_dataset("commonsense_qa")