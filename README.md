# Finetune (Small) Large Language Models on Q/A Tasks

This repository contains code to finetune large language models (GPT-2) on Q/A tasks. It is based on the Huggingface Transformers.

### Installation

``bash install.sh``

### Usage

``python ft_mcq.py --help``

### Example

`python ft_mcq.py --model_name gpt2 --dataset piqa`

### Data

The data is in the following format:

``{"choices": ["Pour it onto a plate", "Pour it into a jar"], "query": "Question: When boiling butter, when it's ready, you can\n", "gold": 1}``

Pre-processed datasets are available in the data folder `datasets`


### Evaluation

``python eval_mcq.py --model_type gpt2 --model_name_or_path output/mcq/ --eval_data_file data/mcq/test.txt --per_gpu_eval_batch_size 1 --max_seq_length 512 --output_dir output/mcq/``

### Why another repo?

Large language models are typically evaluated in zero- or few-shot settings on Q/A and MCQ tasks. But smaller (less than 1B parameters) models give close to random performance on these tasks. This repo contains code to finetune large language models on Q/A tasks. It wraps popular datasets into a fixed MCQ format, and also evaluates the models on these tasks.