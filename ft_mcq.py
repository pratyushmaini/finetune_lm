from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import Dataset
import json, torch
from params import parse_args

def process_data_to_model_inputs(tokenizer):
    def tokenize_function(batch):
        # Tokenize the inputs and labels: prompt contains the question and the choices
        # response contains the correct choice
        # we have to append the response to the prompt. but labels are only for the response: and -1 for the prompt
        num_prompt_tokens = [len(tokenizer.tokenize(batch["prompt"][i])) - 1 for i in range(len(batch["prompt"]))]
        true_input = [batch["prompt"][i] + batch["response"][i] for i in range(len(batch["prompt"]))]
        inputs = tokenizer(true_input, padding="max_length", truncation=True, max_length=900)
        labels = inputs["input_ids"]
        labels = [[-100] * num_prompt_tokens[i] + labels[i][num_prompt_tokens[i]:] for i in range(len(labels))]
        
        # change labels to -100 wherever we have padding
        labels = [[-100 if token == tokenizer.pad_token_id else token for token in label] for label in labels]
        inputs["labels"] = labels

        #confirm that the labels at not -100 are same as the response when decoded do for each ([label for label in labels[0] if label!=-100])
        # real_labels = [[label for label in labels_i if label!=-100] for labels_i in labels]
        # real_labels = [tokenizer.decode(label) for label in real_labels]
        # real_responses = [" " + response for response in batch["response"]]
        # assert real_labels == real_responses, "labels and responses don't match"

        #assert that shape of inputs["input_ids"] is **1024
        assert len(inputs["input_ids"][0]) == 900, "input_ids not of length 1024"
    
        return inputs
    return tokenize_function


def load_data(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        data = [json.loads(line) for line in lines]

    processed_data = {"prompt": [], "response": []}
    from eval_mcq import multiple_choice
    for row in data:
        mc_data = multiple_choice(row)
        processed_data["prompt"].append(mc_data["prompt"])
        processed_data["response"].append(mc_data["response"])
    return processed_data

def my_trainer(args):
    file_path = f"datasets/{args.dataset}/train.jsonl"
    raw_data = load_data(file_path)
    dataset = Dataset.from_dict(raw_data)

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path).cuda()
    tokenizer.pad_token = tokenizer.eos_token


    # Tokenize dataset
    tokenized_dataset = dataset.map(process_data_to_model_inputs(tokenizer), batched=True)

    num_devices = torch.cuda.device_count()
    assert args.total_batch_size % num_devices == 0, "Batch size not divisible by number of devices"
    gradient_accumulation_steps = args.total_batch_size // num_devices // args.per_device_train_batch_size

    #save at every 1/5 of total steps
    num_steps = len(tokenized_dataset) // args.total_batch_size * args.num_epochs
    save_steps = num_steps // args.num_save_steps
    warmup_steps = num_steps // args.warmup_ratio

    #output log and save dir paths
    output_dir = f"models/{args.dataset}"


    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=args.weight_decay,
        logging_dir=output_dir,
        logging_steps=args.logging_steps,
        report_to=args.report_to,
        save_strategy=args.save_strategy,
        save_steps=save_steps,

    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    # Train the model
    trainer.train()

    # Save the model and the tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    parser = parse_args()
    args = parser.parse_args()
    my_trainer(args)