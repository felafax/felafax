# Standard library imports
import os
import sys
import pdb
import json
import time
import shutil
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

# Third-party imports
from absl import app, flags

import jax

jax.distributed.initialize()

import jax.numpy as jnp
import chex
import optax
from datasets import load_dataset
from huggingface_hub import snapshot_download
from transformers import default_data_collator

# Add the current directory and its parent to the Python path.
# This allows importing modules from these directories.
sys.path.append(os.path.abspath(os.getcwd()))
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))

# Local imports
import llama3_jax
from llama3_jax.trainer_engine import setup
from llama3_jax.trainer_engine import (automodel_lib, checkpoint_lib,
                                       convert_lib, jax_utils, llama_config,
                                       trainer_lib, utils, dataset_lib)

FLAGS = flags.FLAGS
flags.DEFINE_string("base_dir", "/home/felafax-storage/llama3_jax/",
                    "Base directory for data")
flags.DEFINE_string("model_export_dir", "/home/felafax-storage/checkpoints/",
                    "Base directory for Hugging Face compatible model export.")
flags.DEFINE_string("model_name", "llama-3.1-8B-Instruct-JAX", "Model name")
flags.DEFINE_string(
    "data_source", None,
    "Path to local JSON data file or Hugging Face dataset name")

flags.DEFINE_boolean("train", False, "Run training and save checkpoint")
flags.DEFINE_boolean("export", False, "Export and convert model")
flags.DEFINE_boolean("train_and_export", False,
                     "Run training, save checkpoint, and export model")

flags.DEFINE_boolean("upload_to_hf", False,
                     "Upload checkpoint to Hugging Face")
flags.DEFINE_boolean("test_dataset", False, "Run dataset pipeline test")

flags.DEFINE_string("hf_token", None, "Hugging Face API token")
flags.DEFINE_string("hf_username", None, "Hugging Face username")
flags.DEFINE_string("hf_repo_name", None, "Hugging Face repository name")

flags.DEFINE_boolean("timeit", False, "Time the run")

flags.DEFINE_string("trainer_config_json", None,
                    "Path to JSON file containing trainer configuration")

flags.DEFINE_string("final_checkpoint_path", None,
                    "Path to the final checkpoint")


@chex.dataclass(frozen=True)
class TrainerConfig:
    learning_rate: float = 1e-3
    num_epochs: int = 1
    max_steps: int | None = 20
    batch_size: int = 16
    seq_length: int = 64
    dataset_size_limit: int | None = None
    print_every_n_steps: int = 5
    eval_every_n_steps: int = 1000
    max_eval_steps: int | None = 1

    @classmethod
    def from_json(cls, json_path: str):
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)


def train_and_save_checkpoint(*, model_name, model_path, model,
                              model_configurator, tokenizer, trainer_config,
                              checkpoint_path, data_source):
    optimizer = optax.sgd(trainer_config.learning_rate)

    dataset = dataset_lib.Dataset(tokenizer)
    train_dataloader, val_dataloader = dataset.get_dataset(
        data_source=data_source,
        batch_size=trainer_config.batch_size,
        seq_length=trainer_config.seq_length,
        max_examples=trainer_config.dataset_size_limit)

    # Print training information
    trainer_lib.pprint_training_pipeline(train_dataloader, trainer_config)

    trainer = trainer_lib.CausalLMTrainer(
        model=model,
        model_ckpt_path=model_path,
        model_configurator=model_configurator,
        optimizer=optimizer,
        training_config=trainer_config,
        mesh=jax_utils.MESH,
        model_name=model_name,
        dtype=jnp.bfloat16,
    )

    start_time = time.time()
    if FLAGS.timeit:
        print(f"Start time: {start_time:.4f}")

    state = trainer.train(train_dataloader, val_dataloader, run_jitted=True)

    if FLAGS.timeit:
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"End time: {end_time:.4f}")
        print(f"Execution time: {elapsed_time:.4f} seconds")

    # Convert global arrays to host-local arrays
    host_local_state = jax.tree_map(
        lambda x: jax.experimental.multihost_utils.
        global_array_to_host_local_array(x, jax_utils.MESH,
                                         jax.sharding.PartitionSpec())
        if isinstance(x, jax.Array) else x, state)

    # Only save on process 0
    if jax.process_index() == 0:
        trainer.save_checkpoint(host_local_state, path=checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    # Ensure all processes are synchronized
    jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()
    print(f"All done! Training and saving checkpoint completed.")


def export_and_convert(
    *,
    model_name,
    model_configurator,
    checkpoint_path,
    hf_export_dir,
    model_export_dir,
):
    print("Checkpoint path:", checkpoint_path)
    convert_lib.save_hf_compatible_checkpoint(
        f'flax_params::{checkpoint_path}', hf_export_dir,
        model_configurator)

    # Download and save the tokenizer
    tokenizer_repo = "felafax/tokenizer-llama-3.1-8B-Instruct-JAX" # TODO: Fix
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

    checkpoint_lib.copy_directory(hf_export_dir, model_export_dir)
    print(f"Checkpoint copied to {model_export_dir}")


def upload_to_huggingface(*, hf_export_dir, hf_username, hf_repo_name,
                          hf_token):
    """Upload checkpoint to Hugging Face."""
    convert_lib.upload_checkpoint_to_hf(hf_export_dir,
                                        f"{hf_username}/{hf_repo_name}",
                                        hf_token)
    print(f"Checkpoint uploaded to Hugging Face: {hf_username}/{hf_repo_name}")


def main(argv):
    del argv  # Unused
    if not FLAGS.data_source:
        raise ValueError("--data_source must be provided")

    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Base dir is where JAX cache and HF HUB cache (so model downloads) are stored.
    os.makedirs(FLAGS.base_dir, exist_ok=True)
    setup.setup_environment(base_dir=FLAGS.base_dir)
    setup.reload_modules("llama3_jax")

    model_path, model, model_configurator, tokenizer = (
        automodel_lib.AutoJAXModelForCausalLM.from_pretrained(
            FLAGS.model_name,
            dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16,
            lora_rank=8,
            lora_alpha=16,
        ))

    # Initialize TrainerConfig
    if FLAGS.trainer_config_json:
        trainer_config = TrainerConfig.from_json(FLAGS.trainer_config_json)
    else:
        trainer_config = TrainerConfig()

    # Checkpoint dir is where Flax checkpoints are saved.
    checkpoint_dir = os.path.join(FLAGS.base_dir,
                                  f"{current_datetime}")
    checkpoint_path = os.path.join(checkpoint_dir, FLAGS.model_name)

    # Temp directory to save Hugging Face compatible export.
    temp_dir = os.path.join(FLAGS.base_dir,
                            f"temp_{current_datetime}")

    model_export_dir = os.path.join(FLAGS.model_export_dir,
                                    f"{current_datetime}")

    print("Base dir:", FLAGS.base_dir)
    print("Model name:", FLAGS.model_name)
    print("Checkpoint dir:", checkpoint_dir)
    print("Checkpoint path:", checkpoint_path)
    print("Temp dir:", temp_dir)
    print("Model export dir:", model_export_dir)


    # Create necessary directories
    utils.makedirs(checkpoint_dir, exist_ok=True)
    utils.makedirs(temp_dir, exist_ok=True)
    utils.makedirs(model_export_dir, exist_ok=True)

    if FLAGS.train or FLAGS.train_and_export:
        train_and_save_checkpoint(
            model_name=FLAGS.model_name,
            model_path=model_path,
            model=model,
            model_configurator=model_configurator,
            tokenizer=tokenizer,
            trainer_config=trainer_config,
            checkpoint_path=checkpoint_path,
            data_source=FLAGS.data_source,
        )

    # Checkpoint will only be saved by process 0, so export will only be done on process 0.
    if jax.process_index() == 0 and (FLAGS.export or FLAGS.train_and_export):
        export_and_convert(
            model_name=FLAGS.model_name,
            model_configurator=model_configurator,
            checkpoint_path=checkpoint_path,
            hf_export_dir=temp_dir,
            model_export_dir=model_export_dir,
        )

    if jax.process_index() == 0 and FLAGS.upload_to_hf:
        if not all([FLAGS.hf_token, FLAGS.hf_username, FLAGS.hf_repo_name]):
            raise ValueError(
                "Hugging Face credentials are required for upload.")
        upload_to_huggingface(hf_export_dir=model_export_dir,
                              hf_username=FLAGS.hf_username,
                              hf_repo_name=FLAGS.hf_repo_name,
                              hf_token=FLAGS.hf_token)

    if FLAGS.test_dataset:
        dataset_lib.test_dataset_pipeline(tokenizer, FLAGS.data_source)
        return


if __name__ == "__main__":
    app.run(main)
