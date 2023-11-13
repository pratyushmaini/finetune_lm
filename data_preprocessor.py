from datasets import load_dataset
import json
# each dataset should be re-written as a jsonl file with the following format:
# In case of MCQ the format should be as follows:
# {"choices": ["Sunlight is the source of energy for nearly all ecosystems.", "Most ecosystems are found on land instead of in water.", "Carbon dioxide is more available than other gases.", "The producers in all ecosystems are plants."], "query": "Question: Which statement best explains why photosynthesis is the foundation of most food webs?\n", "gold": 0}

# In case of QA the format should be as follows:
# {"query": "Question: Which statement best explains why photosynthesis is the foundation of most food webs?\n", "gold": "Sunlight is the source of energy for nearly all ecosystems."}


def commonsense_qa_preprocess(dataset):
    # {'id': '61fe6e879ff18686d7552425a36344c8', 'question': 'Sammy wanted to go to where the people were.  Where might he go?', 'question_concept': 'people', 'choices': {'label': ['A', 'B', 'C', 'D', 'E'], 'text': ['race track', 'populated areas', 'the desert', 'apartment', 'roadblock']}, 'answerKey': 'B'}
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
                "query": f"Question: {question}\n",
                "gold": answerKey
            }
            jsonl.append(jsonl_record)
        jsonl_dict[split] = jsonl
    return jsonl_dict


def piqa_preprocess(dataset):
    # {'goal': 'to be a good student', 'sol1': 'study hard', 'sol2': 'play video games', 'label': 0}
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
                "query": f"Question: {question}\n",
                "gold": answerKey
            }
            jsonl.append(jsonl_record)
        jsonl_dict[split] = jsonl
    return jsonl_dict


def arc_easy_preprocess(dataset):
    # we will take a max of 10k examples from the train split
    # "question": Which factor will most likely cause a person to develop a fever?
    # "choices": { "text": [ "a leg muscle relaxing after exercise", "a bacterial population in the bloodstream", "several viral particles on the skin", "carbohydrates being digested in the stomach" ], "label": [ "A", "B", "C", "D" ] }
    # "answerKey": "B
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
                "query": f"Question: {question}\n",
                "gold": answerKey
            }
            jsonl.append(jsonl_record)
        jsonl_dict[split] = jsonl


preprocess_func = {
    # mcq datasets
    "commonsense_qa": commonsense_qa_preprocess,   
    "piqa": piqa_preprocess,
    "arc_easy": arc_easy_preprocess,
}

def get_dataset(dataset_name):
    dataset = load_dataset(dataset_name, streaming=True)

    # now preprocess the dataset
    jsonl_dict = preprocess_func[dataset_name](dataset)
    #save the jsonl files 
    for split, jsonl in jsonl_dict.items():
        with open(f"datasets/{dataset_name}/{split}.jsonl", "w", encoding='utf-8') as f:
            data = jsonl
            for entry in data:
                json_record = json.dumps(entry, ensure_ascii=False)
                f.write(json_record + '\n')
    return jsonl 

if __name__ == "__main__":
    get_dataset("piqa")