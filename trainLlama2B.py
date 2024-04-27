
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

def create_llm_prompts(filename, positive_template, negative_template):
  """
  Reads data from a JSON file and creates separate LLM training prompts for positive and negative reviews based on templates.

  Args:
      filename: The path to the JSON file.
      positive_template: A string template for positive reviews.
      negative_template: A string template for negative reviews.

  Returns:
      A tuple containing two lists: positive_prompts and negative_prompts.

  Raises:
      FileNotFoundError: If the file is not found.
      json.JSONDecodeError: If the JSON data is invalid.
  """


  positive_prompts = []
  negative_prompts = []
  for entry in data:
    # Check sentiment from "agree" or "disagree" field (adjust based on your data)
    if "review" in entry:
      review = entry
    elif "agree" in entry:
      filled_template = positive_template.format(**entry)
      positive_prompts.append(filled_template)
    elif "disagree" in entry:
      filled_template = negative_template.format(**entry)
      negative_prompts.append(filled_template)
    else:
      # Handle cases where sentiment is not clear (optional)
      print(f"Skipping entry: {entry['review']}, sentiment unclear")

  return positive_prompts, negative_prompts

dataset = load_dataset('json', data_files='mydata.json')

print(dataset["review"][0])
try:
  filename = "your_file.json"
  positive_template = "This review praises the film. Review: {review}\nSummarize the positive aspects mentioned in the review and provide your analysis supporting those points."
  negative_template = "This review criticizes the film. Review: {review}\nSummarize the negative aspects mentioned in the review and provide your analysis addressing those points."

  positive_prompts, negative_prompts = create_llm_prompts(filename, positive_template, negative_template)

  print("Positive Prompts:")
  for prompt in positive_prompts:
    print(prompt)

  print("\nNegative Prompts:")
  for prompt in negative_prompts:
    print(prompt)
except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
  print(f"Error creating LLM prompts: {e}")

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
dataset = load_dataset("flytech/python-codes-25k", split='train').train_test_split(test_size=.8, train_size=.2)

# Create the trainer
trainer = SFTTrainer(
    model = model,
    # train_dataset = dataset,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    dataset_text_field = "text",
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
    ),
)

# Train the model
trainer.train()

# Save the trained model
trainer.save_model("./Llama")