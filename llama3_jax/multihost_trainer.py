# Standard library imports
import os
import sys
import pdb
import gzip
import shutil
import time
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

# Third-party imports
from absl import app, flags
import chex
import jax
import jax.numpy as jnp
import optax
import torch
from datasets import load_dataset
from huggingface_hub import snapshot_download
from transformers import default_data_collator

# Add the current directory and its parent to the Python path.
# This allows importing modules from these directories.
sys.path.append(os.path.abspath(os.getcwd()))
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))

# Local imports
try:
    import llama3_jax
    print("felafax package imported successfully")
except ImportError as e:
    print(f"Error importing llama3_jax: {e}")

from llama3_jax.trainer_engine import setup
from llama3_jax import llama_config
from llama3_jax.trainer_engine import (automodel_lib, checkpoint_lib,
                                       convert_lib, jax_utils, trainer_lib,
                                       utils)

FLAGS = flags.FLAGS

flags.DEFINE_string("base_dir", "/mnt/persistent-disk",
                    "Base directory for data")
flags.DEFINE_string("model_name", "llama-3.1-8B-Instruct-JAX", "Model name")
flags.DEFINE_boolean("train", False, "Run training and save checkpoint")
flags.DEFINE_boolean("export", False, "Export and convert model")
flags.DEFINE_boolean("test_dataset", False, "Run dataset pipeline test")
flags.DEFINE_boolean("timeit", False, "Time the run")


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
    dataset = load_dataset(
        "json",
        data_files="/home/alpaca-cleaned/alpaca_data_cleaned.json",
        split="train")
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
                                  batch_size=4,
                                  seq_length=32,
                                  max_examples=32)
    batch = next(iter(train_loader))
    print("Input tokens shape:", batch['input_tokens'].shape)
    print("Target mask shape:", batch['target_tokens'].shape)


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


def train_and_save_checkpoint(*, base_dir, model_name, model_path, model,
                              model_configurator, tokenizer, trainer_config):
    optimizer = optax.sgd(trainer_config.learning_rate)

    train_dataloader, val_dataloader = get_dataset(
        tokenizer=tokenizer,
        batch_size=trainer_config.batch_size,
        seq_length=trainer_config.seq_length,
        max_examples=trainer_config.dataset_size_limit,
    )

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
        model_name=model_name,
    )
    start_time = time.time()
    if FLAGS.timeit:
        print(f"Start time: {start_time:.4f}")

    state = trainer.train(train_dataloader, val_dataloader, run_jitted=False)

    if FLAGS.timeit:
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"End time: {end_time:.4f}")
        print(f"Execution time: {elapsed_time:.4f} seconds")

    export_dir = os.path.join(base_dir, "export")
    utils.makedirs(export_dir, exist_ok=True)

    flax_checkpoint_path = os.path.join(export_dir, model_name)
    trainer.save_checkpoint(state, path=flax_checkpoint_path)
    print(f"Checkpoint saved to {flax_checkpoint_path}")


def export_and_convert(*, base_dir, model_name, model_configurator):
    export_dir = os.path.join(base_dir, "export")
    hf_export_dir = os.path.join(base_dir, "hf_export")

    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    gcs_dir = f"/home/felafax-storage/checkpoints/{model_name}/{current_datetime}/"

    utils.makedirs(hf_export_dir, exist_ok=True)
    utils.makedirs(gcs_dir, exist_ok=True)

    flax_checkpoint_path = os.path.join(export_dir, model_name)
    convert_lib.save_hf_compatible_checkpoint(
        f'flax_params::{flax_checkpoint_path}', hf_export_dir,
        model_configurator)

    # Download and save the tokenizer
    tokenizer_repo = f"felafax/tokenizer-{model_name}"
    tokenizer_dir = snapshot_download(repo_id=tokenizer_repo)

    # Move all files from tokenizer_dir to hf_export_dir
    for item in os.listdir(tokenizer_dir):
        s = os.path.join(tokenizer_dir, item)
        d = os.path.join(hf_export_dir, item)
        if os.path.isfile(s):
            shutil.copy2(s, d)
            print(f"Copied {item} to {hf_export_dir}")
        elif os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
            print(f"Copied directory {item} to {hf_export_dir}")
    print(f"All tokenizer files saved to {hf_export_dir}")

    checkpoint_lib.copy_directory(hf_export_dir, gcs_dir)
    print(f"Checkpoint copied to {gcs_dir}")


def main(argv):
    del argv  # Unused

    setup.setup_environment(base_dir=FLAGS.base_dir)
    setup.reload_modules("llama3_jax")

    model_path, model, model_configurator, tokenizer = (
        automodel_lib.AutoJAXModelForCausalLM.from_pretrained(
            FLAGS.model_name))

    trainer_config = TrainerConfig()

    if FLAGS.test_dataset:
        test_dataset_pipeline(tokenizer)
        return

    if FLAGS.train:
        train_and_save_checkpoint(base_dir=FLAGS.base_dir,
                                  model_name=FLAGS.model_name,
                                  model_path=model_path,
                                  model=model,
                                  model_configurator=model_configurator,
                                  tokenizer=tokenizer,
                                  trainer_config=trainer_config)

    if FLAGS.export:
        export_and_convert(base_dir=FLAGS.base_dir,
                           model_name=FLAGS.model_name,
                           model_configurator=model_configurator)


# HUGGINGFACE_TOKEN = input("INPUT: Please provide your HUGGINGFACE_TOKEN: ")
# HUGGINGFACE_USERNAME = input(
#     "INPUT: Please provide your HUGGINGFACE_USERNAME: ")
# HUGGINGFACE_REPO_NAME = input(
#     "INPUT: Please provide your HUGGINGFACE_REPO_NAME: ")
# convert_lib.upload_checkpoint_to_hf(
#     HF_EXPORT_DIR, f"{HUGGINGFACE_USERNAME}/{HUGGINGFACE_REPO_NAME}",
#     HUGGINGFACE_TOKEN)

if __name__ == "__main__":
    app.run(main)
