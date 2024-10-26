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
flags.DEFINE_string("base_dir", "/mnt/persistent-disk",
                    "Base directory for data")
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

flags.DEFINE_boolean("download_model", False, "Download the model on process index 0")
flags.DEFINE_string("final_checkpoint_path", None, "Path to the final checkpoint")

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
                              flax_checkpoint_path, data_source):
    optimizer = optax.sgd(trainer_config.learning_rate)

    dataset = dataset_lib.Dataset(tokenizer)
    train_dataloader, val_dataloader = dataset.get_dataset(
        data_source=data_source,
        batch_size=trainer_config.batch_size,
        seq_length=trainer_config.seq_length,
        max_examples=trainer_config.dataset_size_limit
    )

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
        global_array_to_host_local_array(x, jax_utils.MESH, jax.sharding.PartitionSpec())
        if isinstance(x, jax.Array) else x, state)

    # Only save on process 0
    if jax.process_index() == 0:
        trainer.save_checkpoint(host_local_state, path=flax_checkpoint_path)
        print(f"Checkpoint saved to {flax_checkpoint_path}")

    # Ensure all processes are synchronized
    jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()
    print(f"Checkpoint saved to {flax_checkpoint_path}")


def export_and_convert(
    *,
    model_name,
    model_configurator,
    flax_checkpoint_path,
    hf_export_dir,
    gcs_dir,
):
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


def upload_to_huggingface(*, hf_export_dir, hf_username, hf_repo_name,
                          hf_token):
    """Upload checkpoint to Hugging Face."""
    convert_lib.upload_checkpoint_to_hf(hf_export_dir,
                                        f"{hf_username}/{hf_repo_name}",
                                        hf_token)
    print(f"Checkpoint uploaded to Hugging Face: {hf_username}/{hf_repo_name}")


def download_model(model_name):
    print(f"Downloading model {model_name} on process 0...")
    model_path, model, model_configurator, tokenizer = (
        automodel_lib.AutoJAXModelForCausalLM.from_pretrained(
            model_name,
            dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16,
            lora_rank=8,
            lora_alpha=16,
        )
    )
    print("Model download complete.")
    return model_path, model, model_configurator, tokenizer


def main(argv):
    del argv  # Unused
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # check if base_dir is not a directory
    if not os.path.isdir(FLAGS.base_dir):
        raise ValueError(f"Base directory {FLAGS.base_dir} is not a directory")

    FLAGS.base_dir = f"{FLAGS.base_dir}/{current_datetime}"
    os.makedirs(FLAGS.base_dir, exist_ok=True)

    setup.setup_environment(base_dir=FLAGS.base_dir)
    setup.reload_modules("llama3_jax")

    if FLAGS.download_model is True:
        model_path, model, model_configurator, tokenizer = download_model(FLAGS.model_name)
    else:
        model_path, model, model_configurator, tokenizer = (
            automodel_lib.AutoJAXModelForCausalLM.from_pretrained(
                FLAGS.model_name,
                dtype=jnp.bfloat16,
                param_dtype=jnp.bfloat16,
                lora_rank=8,
                lora_alpha=16,
            )
        )

    # Initialize TrainerConfig
    if FLAGS.trainer_config_json:
        trainer_config = TrainerConfig.from_json(FLAGS.trainer_config_json)
    else:
        trainer_config = TrainerConfig()

    # Define directories and paths
    export_dir = os.path.join(FLAGS.base_dir, "export")
    hf_export_dir = os.path.join(FLAGS.base_dir, "hf_export")

    if FLAGS.final_checkpoint_path is not None:
        final_checkpoint_path = FLAGS.final_checkpoint_path
    else:
        # TODO: remove the hard-coded path for checkpoint
        final_checkpoint_path = (f"/home/felafax-storage/checkpoints/{FLAGS.model_name}/"
               f"{current_datetime}/")


    flax_checkpoint_path = os.path.join(export_dir, FLAGS.model_name)

    # Create necessary directories
    utils.makedirs(export_dir, exist_ok=True)
    utils.makedirs(hf_export_dir, exist_ok=True)
    utils.makedirs(final_checkpoint_path, exist_ok=True)

    if not FLAGS.data_source:
        raise ValueError("--data_source must be provided")

    if FLAGS.train or FLAGS.train_and_export:
        train_and_save_checkpoint(
            model_name=FLAGS.model_name,
            model_path=model_path,
            model=model,
            model_configurator=model_configurator,
            tokenizer=tokenizer,
            trainer_config=trainer_config,
            flax_checkpoint_path=flax_checkpoint_path,
            data_source=FLAGS.data_source,
        )

    if FLAGS.export or FLAGS.train_and_export:
        export_and_convert(
            model_name=FLAGS.model_name,
            model_configurator=model_configurator,
            flax_checkpoint_path=flax_checkpoint_path,
            hf_export_dir=hf_export_dir,
            gcs_dir=final_checkpoint_path,
        )

    if FLAGS.upload_to_hf:
        if not all([FLAGS.hf_token, FLAGS.hf_username, FLAGS.hf_repo_name]):
            raise ValueError(
                "Hugging Face credentials are required for upload.")
        upload_to_huggingface(hf_export_dir=hf_export_dir,
                              hf_username=FLAGS.hf_username,
                              hf_repo_name=FLAGS.hf_repo_name,
                              hf_token=FLAGS.hf_token)

    if FLAGS.test_dataset:
        dataset_lib.test_dataset_pipeline(tokenizer, FLAGS.data_source)
        return


if __name__ == "__main__":
    app.run(main)
