
import os
from dataclasses import dataclass, field
from typing import Optional
from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from tqdm.notebook import tqdm

from trl import SFTTrainer
from accelerate import Accelerator

import json

def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['review'])):
        text = f"### Here is a review: {example['review'][i]}\n ### Tell me why you agree with the review: {example['agree'][i]}"
        output_texts.append(text)
        text = f"### Here is a review: {example['review'][i]}\n ### Tell me why you disagree with the review: {example['disagree'][i]}"
        output_texts.append(text)
    return output_texts

dataset = load_dataset('csv', data_files='mycsvdata.csv')

accelerator = Accelerator()

# interpreter_login()

torch.cuda.empty_cache() 

# compute_dtype = getattr(torch, "float16")

device_map = "auto"
max_seq_length = 2048 # Supports RoPE Scaling interally, so choose any!
# Download model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "./LlamaBase", # Supports Llama, Mistral - replace this!
    max_seq_length = max_seq_length,
    dtype = None,
    load_in_4bit = True,
)


# Do model patching and add fast LoRA weights
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    use_gradient_checkpointing = True,
    random_state = 3407,
    max_seq_length = max_seq_length,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)


# Load the dataset
# dataset = load_dataset("flytech/python-codes-25k", split='train').train_test_split(test_size=.8, train_size=.2)

# Create the trainer
trainer = SFTTrainer(
    model = model,
    # train_dataset = dataset,
    train_dataset=dataset['train'],
    # dataset_text_field = "text",
    formatting_func=formatting_prompts_func,
    max_seq_length = max_seq_length,
    tokenizer = tokenizer,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        # max_steps = 1000,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 100,
        output_dir = "outputs",
        optim = "adamw_8bit",
        seed = 1222,
        num_train_epochs=10
    ),
)

# Train the model
trainer.train()

# Save the trained model
trainer.save_model("./Llama2b")