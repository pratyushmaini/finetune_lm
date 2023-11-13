from data_module import TextDatasetQA
from dataloader import CustomTrainer, custom_data_collator
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import os
from pathlib import Path
import json 
from tqdm import tqdm
from omegaconf import OmegaConf

def main(cfg):
    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        device_map = {'': local_rank}

    os.environ["WANDB_DISABLED"] = "true"

    Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)
    #save the cfg file
    with open(f'{cfg.save_dir}/cfg.yaml', 'w') as f:
        OmegaConf.save(cfg, f)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    tokenizer.pad_token = tokenizer.eos_token

    max_length = 1024
    torch_format_dataset = TextDatasetQA(cfg.data_path, tokenizer=tokenizer, max_length=max_length, split=cfg.split)

    batch_size = 16
    gradient_accumulation_steps = 1
    steps_per_epoch = len(torch_format_dataset)//(batch_size*gradient_accumulation_steps*torch.cuda.device_count())

    max_steps = int(cfg.num_epochs*len(torch_format_dataset))//(batch_size*gradient_accumulation_steps*torch.cuda.device_count())

    training_args = transformers.TrainingArguments(
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=max(1, max_steps//10),
            max_steps=max_steps,
            learning_rate=cfg.lr,
            logging_steps=max(1,max_steps//20),
            logging_dir=f'{cfg.save_dir}/logs',
            output_dir=cfg.save_dir,
            optim="paged_adamw_32bit",
            save_steps=steps_per_epoch,
            ddp_find_unused_parameters= False,
            evaluation_strategy="no",
        )

    model = AutoModelForCausalLM.from_pretrained(cfg.model_id, use_flash_attention_2=True, device_map=device_map)

    trainer = CustomTrainer(
        model=model,
        train_dataset=torch_format_dataset,
        eval_dataset=torch_format_dataset,
        args=training_args,
        data_collator=custom_data_collator,
    )
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    trainer.train()

    #save the model
    model = model.merge_and_unload()



    # trainer.save_model(cfg.save_dir + "trainer")
    #save the tokenizer
    tokenizer.save_pretrained(cfg.save_dir)
    model.save_pretrained(cfg.save_dir)

if __name__ == "__main__":
    main()