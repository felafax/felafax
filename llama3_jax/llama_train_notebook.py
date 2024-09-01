#!/usr/bin/env python
# coding: utf-8

import os
import sys
import importlib

# Add the parent directory of the current working directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

try:
    import felafax
    print("felafax package imported successfully")
except ImportError as e:
    print(f"Error importing felafax: {e}")

from felafax.trainer_engine import setup

setup.setup_environment()

from felafax.trainer_engine import utils, jax_utils
from felafax.trainer_engine import automodel_lib, checkpoint_lib, trainer_lib, convert_lib
from felafax import llama_config

setup.reload_modules()

from typing import (Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union)
import jax
import jax.numpy as jnp
import chex
import optax
import torch
from datasets import load_dataset
from transformers import default_data_collator

HUGGINGFACE_USERNAME = input(
    "INPUT: Please provide your HUGGINGFACE_USERNAME: ") or "felarof01"
HUGGINGFACE_TOKEN = input("INPUT: Please provide your HUGGINGFACE_TOKEN: "
                          ) or "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"

# Select a supported model from above list to use!
MODEL_NAME = "Meta-Llama-3.1-8B"

# Constants for paths
FELAFAX_DIR = os.path.dirname(os.path.dirname(felafax.__file__))
GCS_DIR = "/home/felafax-storage/"
EXPORT_DIR = os.path.join(FELAFAX_DIR, "export")
HF_COMPATIBLE_EXPORT_DIR = os.path.join(GCS_DIR, "hf_export")
HF_REPO_ID = "felarof01/test_checkpoint"

# Ensure directories exist
utils.makedirs(EXPORT_DIR, exist_ok=True)
utils.makedirs(HF_COMPATIBLE_EXPORT_DIR, exist_ok=True)

model_path, model, model_configurator, tokenizer = automodel_lib.AutoJAXModelForCausalLM.from_pretrained(
    "llama-3.1-8B-JAX", HUGGINGFACE_TOKEN)


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
    max_steps: int | None = 5
    batch_size: int = 32
    seq_length: int = 64
    dataset_size_limit: int | None = 32
    print_every_n_steps: int = 1
    eval_every_n_steps: int = 1000


training_cfg = TrainingConfig()
optimizer = optax.sgd(training_cfg.learning_rate)

# Prepare dataset
train_dataloader, val_dataloader = get_dataset(
    tokenizer=tokenizer,
    seq_length=training_cfg.seq_length,
    max_examples=training_cfg.dataset_size_limit,
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

export_path = os.path.join(EXPORT_DIR, "llama3.flax")
trainer.save_checkpoint(state, path=export_path)

convert_lib.save_hf_compatible_checkpoint(f'flax_params::{export_path}',
                                          HF_COMPATIBLE_EXPORT_DIR,
                                          model_configurator)

# convert_lib.upload_checkpoint_to_hf(HF_COMPATIBLE_EXPORT_DIR, HF_REPO_ID, HUGGINGFACE_TOKEN)
