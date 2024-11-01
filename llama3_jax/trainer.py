# Standard library imports
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

# Third-party imports
from absl import app, flags
import chex
import jax
import jax.numpy as jnp
import optax
from datasets import load_dataset
from flax.training import train_state
from huggingface_hub import snapshot_download
from jax.experimental import multihost_utils
from jax.sharding import PartitionSpec as PS
from transformers import default_data_collator

# Add the current directory and its parent to the Python path
sys.path.append(os.path.abspath(os.getcwd()))
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))

# Local imports
import llama3_jax
from llama3_jax.trainer_engine import (
    automodel_lib,
    checkpoint_lib,
    convert_lib,
    dataset_lib,
    jax_utils,
    llama_config,
    setup,
    trainer_lib,
    utils,
)

FLAGS = flags.FLAGS
flags.DEFINE_string("base_dir", None, "Base directory for data")
flags.DEFINE_string("model_name", None, "Model name")
flags.DEFINE_string("data_source", None, "Path to dataset")
flags.DEFINE_boolean("train", False, "Run training")

class TrainerConfig:
    def __init__(self):
        self.learning_rate = 1e-3
        self.num_epochs = 1
        self.max_steps = 20
        self.batch_size = 16
        self.seq_length = 64
        self.print_every_n_steps = 5

def train_step(state, batch, rng, model):
    """Single training step."""
    def compute_loss(params, lora_params):
        variables = {'params': params, 'lora_params': lora_params}
        logits = model.apply(
            variables,
            batch["input_tokens"],
            deterministic=False,
            rngs={'dropout': rng},
        ).logits
        return jax_utils.cross_entropy_loss_and_accuracy(
            logits, 
            batch["target_tokens"], 
            batch["loss_masks"]
        )

    (loss, accuracy), grads = jax.value_and_grad(
        compute_loss, argnums=1, has_aux=True
    )(state.params, state.lora_params)

    # Update parameters
    updates, new_opt_state = state.tx.update(
        grads, state.opt_state, state.lora_params
    )
    new_lora_params = optax.apply_updates(state.lora_params, updates)

    new_state = state.replace(
        step=state.step + 1,
        lora_params=new_lora_params,
        opt_state=new_opt_state
    )

    metrics = {"loss": loss, "accuracy": accuracy}
    return new_state, metrics

def train(
    model,
    train_dataloader,
    training_config,
    mesh,
):
    """Main training function."""
    
    # Initialize optimizer
    optimizer = optax.sgd(training_config.learning_rate)

    # Initialize model state
    def init_model():
        rng = jax.random.PRNGKey(0)
        params = model.init(
            input_ids=jnp.zeros((4, training_config.seq_length), dtype=jnp.int32),
            position_ids=jnp.zeros((4, training_config.seq_length), dtype=jnp.int32),
            attention_mask=jnp.ones((4, training_config.seq_length), dtype=jnp.int32),
            rngs={'dropout': rng, 'params': rng},
        )
        return train_state.TrainState.create(
            apply_fn=model.apply,
            params=params['params'],
            lora_params=params['lora_params'],
            tx=optimizer
        )

    # JIT compile training step
    jitted_train_step = jax.jit(train_step)

    # Initialize training state
    with mesh:
        train_state = init_model()
        rng = jax.random.PRNGKey(0)

        # Training loop
        for epoch in range(training_config.num_epochs):
            print(f"Starting epoch {epoch}")
            
            for step, batch in enumerate(train_dataloader):
                if training_config.max_steps and step >= training_config.max_steps:
                    break

                # Prepare batch for distributed training
                batch = multihost_utils.host_local_array_to_global_array(
                    batch, mesh, PS("dp", "fsdp")
                )

                # Training step
                rng, step_rng = jax.random.split(rng)
                train_state, metrics = jitted_train_step(
                    train_state, batch, step_rng, model
                )

                # Logging
                if step % training_config.print_every_n_steps == 0:
                    print(
                        f"Step {step}, "
                        f"Loss: {metrics['loss']:.4f}, "
                        f"Accuracy: {metrics['accuracy']:.4f}"
                    )

        return train_state

def main(argv):
    del argv  # Unused
    if not FLAGS.data_source:
        raise ValueError("--data_source must be provided")

    # Initialize JAX distributed
    # jax.distributed.initialize()

    # Load model and tokenizer
    model_path, model, model_configurator, tokenizer = (
        automodel_lib.AutoJAXModelForCausalLM.from_pretrained(
            FLAGS.model_name,
            dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16,
            lora_rank=8,
            lora_alpha=16,
        ))

    # Initialize config
    training_config = TrainerConfig()

    # Setup dataset
    dataset = dataset_lib.Dataset(tokenizer)
    train_dataloader, _ = dataset.get_dataset(
        data_source=FLAGS.data_source,
        batch_size=training_config.batch_size,
        seq_length=training_config.seq_length,
    )

    if FLAGS.train:
        train_state = train(
            model=model,
            train_dataloader=train_dataloader,
            training_config=training_config,
            mesh=jax_utils.MESH,
        )

if __name__ == "__main__":
    app.run(main)
