#!/usr/bin/env python
# coding: utf-8

import importlib
import os
import sys

import jax
import jax.numpy as jnp

# Add the current directory and its parent to the Python path.
# This allows importing modules from these directories.
sys.path.append(os.path.abspath(os.getcwd()))
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))

try:
    import llama3_jax
    print("felafax package imported successfully")
except ImportError as e:
    print(f"Error importing llama3_jax: {e}")

from llama3_jax.trainer_engine import setup

setup.setup_environment(base_dir="/mnt/persistent-disk/")

from llama3_jax import llama_config
from llama3_jax.trainer_engine import (automodel_lib, checkpoint_lib,
                                       convert_lib, jax_utils, trainer_lib,
                                       utils)

setup.reload_modules("llama3_jax")

import gzip
import os
import shutil
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import chex
import jax
import jax.numpy as jnp
import optax
import torch
from datasets import load_dataset
from huggingface_hub import snapshot_download
from transformers import default_data_collator

# Select a supported model from above list to use!
MODEL_NAME = "llama-3.1-8B-Instruct-JAX"

# Constants for paths
FELAFAX_DIR = "/mnt/persistent-disk" # os.path.dirname(os.path.dirname(llama3_jax.__file__))

EXPORT_DIR = os.path.join(FELAFAX_DIR, "export")
HF_EXPORT_DIR = os.path.join(FELAFAX_DIR, "hf_export")

current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
GCS_DIR = f"/home/felafax-storage/checkpoints/{MODEL_NAME}/{current_datetime}/"

# Ensure directories exist
utils.makedirs(EXPORT_DIR, exist_ok=True)
utils.makedirs(HF_EXPORT_DIR, exist_ok=True)
utils.makedirs(GCS_DIR, exist_ok=True)

model_path, model, model_configurator, tokenizer = (
    automodel_lib.AutoJAXModelForCausalLM.from_pretrained(MODEL_NAME))


def get_dataset(*, tokenizer, batch_size=1, seq_length=32, max_examples=None):
    # Define Alpaca prompt template
    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
    
    ### Instruction: {}
    
    ### Input: {}
    
    ### Response: {}"""

    EOS_TOKEN = tokenizer.eos_token

    # Defines formatting function.
    def _format_prompts(examples):
        instructions = examples["instruction"]
        inputs = examples["input"]
        outputs = examples["output"]
        texts = []
        for instruction, input, output in zip(instructions, inputs, outputs):
            text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
            texts.append(text)
        return {"text": texts}

    def _tokenize(examples):
        tokenized = tokenizer(examples["text"],
                              truncation=True,
                              padding="max_length",
                              max_length=seq_length + 1)
        return {
            'input_tokens':
            [input_id[:-1] for input_id in tokenized['input_ids']],
            'target_tokens':
            [input_id[1:] for input_id in tokenized['input_ids']],
            'loss_masks':
            [input_id[1:] for input_id in tokenized['attention_mask']]
        }

    def _custom_collate_fn(
            batch: List[Dict[str, Any]]) -> Dict[str, jnp.ndarray]:
        """
        Collates batch items and converts PyTorch tensors to JAX arrays.
        Applies default_data_collator, then converts tensors to JAX format.
        """
        collated = default_data_collator(batch)
        jax_batch = {}
        for key, value in collated.items():
            jax_batch[key] = jnp.array(value.numpy()) if isinstance(
                value, torch.Tensor) else value

        return jax_batch

    # Load and preprocess the dataset
    dataset = load_dataset("yahma/alpaca-cleaned", split="train")
    if max_examples:
        dataset = dataset.select(range(max_examples))
    dataset = dataset.map(_format_prompts, batched=True)

    # Create train and test dataset.
    ds = dataset.train_test_split(test_size=0.15)
    for split in ['train', 'test']:
        ds[split] = ds[split].map(_tokenize,
                                  batched=True,
                                  remove_columns=dataset.column_names)

    # Create DataLoaders
    dataloader_args = dict(shuffle=True,
                           batch_size=batch_size,
                           collate_fn=_custom_collate_fn)
    train_dataloader = torch.utils.data.DataLoader(ds['train'],
                                                   **dataloader_args)
    test_dataloader = torch.utils.data.DataLoader(ds['test'],
                                                  **dataloader_args)

    return train_dataloader, test_dataloader


def test_dataset_pipeline(tokenizer):
    """Print shapes of first batch to verify dataset pipeline."""
    train_loader, _ = get_dataset(tokenizer=tokenizer,
                                  batch_size=1,
                                  seq_length=32,
                                  max_examples=32)
    batch = next(iter(train_loader))
    print("Input tokens shape:", batch['input_tokens'].shape)
    print("Target mask shape:", batch['target_tokens'].shape)


test_dataset_pipeline(tokenizer)


@chex.dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float = 1e-4
    num_epochs: int = 1
    max_steps: int | None = 1
    batch_size: int = 32
    seq_length: int = 64
    dataset_size_limit: int | None = 32
    print_every_n_steps: int = 1
    eval_every_n_steps: int = 1000
    max_eval_steps: int | None = 1


training_cfg = TrainingConfig()
optimizer = optax.sgd(training_cfg.learning_rate)

# Prepare dataset
train_dataloader, val_dataloader = get_dataset(
    tokenizer=tokenizer,
    seq_length=training_cfg.seq_length,
    max_examples=training_cfg.dataset_size_limit,
)

# Calculate and print training steps information
total_samples = len(train_dataloader.dataset)
batch_size = training_cfg.batch_size
steps_per_epoch = (total_samples + batch_size - 1) // batch_size
total_steps = steps_per_epoch * training_cfg.num_epochs

if training_cfg.max_steps:
    total_steps = min(total_steps, training_cfg.max_steps)

print("\nTraining Configuration Summary:")
print(f"Total samples: {total_samples}")
print(f"Batch size: {batch_size}")
print(f"Number of epochs: {training_cfg.num_epochs}")
print(f"Steps per epoch: {steps_per_epoch}")
print(f"Total training steps: {total_steps}")
if training_cfg.max_steps and total_steps == training_cfg.max_steps:
    print(
        f"*Note*: Total steps limited by max_steps setting ({training_cfg.max_steps})"
    )

trainer = trainer_lib.CausalLMTrainer(
    model=model,
    model_ckpt_path=model_path,
    model_configurator=model_configurator,
    optimizer=optimizer,
    training_config=training_cfg,
    mesh=jax_utils.MESH,
)

state = trainer.train(train_dataloader, val_dataloader, run_jitted=True)

flax_checkpoint_path = os.path.join(EXPORT_DIR, MODEL_NAME)
trainer.save_checkpoint(state, path=flax_checkpoint_path)

convert_lib.save_hf_compatible_checkpoint(
    f'flax_params::{flax_checkpoint_path}', HF_EXPORT_DIR, model_configurator)

# Download and save the tokenizer
tokenizer_repo = f"felafax/tokenizer-{MODEL_NAME}"
tokenizer_dir = snapshot_download(repo_id=tokenizer_repo)

# Move all files from tokenizer_dir to HF_EXPORT_DIR
for item in os.listdir(tokenizer_dir):
    s = os.path.join(tokenizer_dir, item)
    d = os.path.join(HF_EXPORT_DIR, item)
    if os.path.isfile(s):
        shutil.copy2(s, d)
        print(f"Copied {item} to {HF_EXPORT_DIR}")
    elif os.path.isdir(s):
        shutil.copytree(s, d, dirs_exist_ok=True)
        print(f"Copied directory {item} to {HF_EXPORT_DIR}")
print(f"All tokenizer files saved to {HF_EXPORT_DIR}")

checkpoint_lib.copy_directory(HF_EXPORT_DIR, GCS_DIR)
print(f"Checkpoint copied to {GCS_DIR}")

# HUGGINGFACE_TOKEN = input("INPUT: Please provide your HUGGINGFACE_TOKEN: ")
# HUGGINGFACE_USERNAME = input(
#     "INPUT: Please provide your HUGGINGFACE_USERNAME: ")
# HUGGINGFACE_REPO_NAME = input(
#     "INPUT: Please provide your HUGGINGFACE_REPO_NAME: ")
# convert_lib.upload_checkpoint_to_hf(
#     HF_EXPORT_DIR, f"{HUGGINGFACE_USERNAME}/{HUGGINGFACE_REPO_NAME}",
#     HUGGINGFACE_TOKEN)
