#!/usr/bin/env python
# coding: utf-8
import importlib
import os
import sys
import pdb

# Add the current directory and its parent to the Python path.
# This allows importing modules from these directories.
sys.path.append(os.path.abspath(os.getcwd()))
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))

import llama3_jax
from llama3_jax.trainer_engine import setup
setup.setup_environment(base_dir="/mnt/persistent-disk")

from llama3_jax.trainer_engine import (automodel_lib, checkpoint_lib,
                                       convert_lib, dataset_lib, jax_utils, llama_config,
                                       trainer_lib, utils)
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

MODEL_NAME = "colab-llama-3.1-8B-Instruct-JAX"
model_path, model, model_configurator, tokenizer = (
    automodel_lib.AutoJAXModelForCausalLM.from_pretrained(MODEL_NAME))


@chex.dataclass(frozen=True)
class TrainerConfig:
    learning_rate: float = 1e-4
    num_epochs: int = 1
    max_steps: int | None = 100
    batch_size: int = 16
    seq_length: int = 64
    dataset_size_limit: int | None = None
    print_every_n_steps: int = 5
    eval_every_n_steps: int = 1000
    max_eval_steps: int | None = 1


trainer_config = TrainerConfig()
optimizer = optax.sgd(trainer_config.learning_rate)

# Prepare dataset
dataset = dataset_lib.Dataset(tokenizer)
train_dataloader, val_dataloader = dataset.get_dataset(
    data_source="yahma/alpaca-cleaned",
    batch_size=trainer_config.batch_size,
    seq_length=trainer_config.seq_length,
    max_examples=trainer_config.dataset_size_limit
)

# Test dataset pipeline
dataset_lib.test_dataset_pipeline(tokenizer, "yahma/alpaca-cleaned")

# Calculate and print training steps information
total_samples = len(train_dataloader.dataset)
batch_size = trainer_config.batch_size
steps_per_epoch = (total_samples + batch_size - 1) // batch_size
total_steps = steps_per_epoch * trainer_config.num_epochs

if trainer_config.max_steps:
    total_steps = min(total_steps, trainer_config.max_steps)

print("\nTraining Configuration Summary:")
print(f"Total samples: {total_samples}")
print(f"Batch size: {batch_size}")
print(f"Number of epochs: {trainer_config.num_epochs}")
print(f"Steps per epoch: {steps_per_epoch}")
print(f"Total training steps: {total_steps}")
if trainer_config.max_steps and total_steps == trainer_config.max_steps:
    print(
        f"*Note*: Total steps limited by max_steps setting ({trainer_config.max_steps})"
    )

trainer = trainer_lib.CausalLMTrainer(
    model=model,
    model_ckpt_path=model_path,
    model_configurator=model_configurator,
    optimizer=optimizer,
    training_config=trainer_config,
    mesh=jax_utils.MESH,
    model_name=MODEL_NAME,
)

state = trainer.train(train_dataloader, val_dataloader, run_jitted=True)


########################################################
# Exporting fine-tuned model
########################################################
# Constants for paths to storage
FELAFAX_DIR = "/mnt/persistent-disk"
EXPORT_DIR = os.path.join(FELAFAX_DIR, "export")
HF_EXPORT_DIR = os.path.join(FELAFAX_DIR, "hf_export")
current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
GCS_DIR = f"/home/felafax-storage/checkpoints/{MODEL_NAME}/{current_datetime}/"

utils.makedirs(EXPORT_DIR, exist_ok=True)
utils.makedirs(HF_EXPORT_DIR, exist_ok=True)
utils.makedirs(GCS_DIR, exist_ok=True)

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
