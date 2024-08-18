# Standard library imports
import os
import pdb
import enum
import re
import string
from dataclasses import dataclass
import functools
from functools import partial
from typing import Any, List, Dict, Tuple, Optional, Union, Sequence, Mapping

# JAX and related libraries (including Flax and Optax)
import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
from flax.training import train_state
from flax.core.meta import unbox
import optax
import chex
import lorax

# JAX model partitioning and sharding
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as PS
from jax.lax import with_sharding_constraint
from jax.experimental import mesh_utils

# Hugging Face Transformers and Datasets
from transformers import (
    AutoModelForCausalLM,
    AutoConfig,
    AutoTokenizer,
    default_data_collator,
)
from datasets import Dataset, load_dataset, concatenate_datasets
import torch

# Gemma-specific imports
from gemma import params as params_lib
from gemma import sampler as sampler_lib
from gemma import transformer as transformer_lib

# Additional imports
from tqdm import tqdm, trange
import numpy as np
import mlxu
from .jax_utils import (
    JaxRNG,
    cross_entropy_loss_and_accuracy,
    global_norm,
    with_sharding_constraint,
)

def forward_and_loss_fn(
    params,
    state,
    batch: Dict[str, Any],
) -> Tuple[jax.Array, Dict[str, Any]]:
    """Forward pass and loss function."""
    input_tokens = batch['input_tokens']
    target_tokens = batch['target_tokens']
    loss_masks = batch['loss_masks']

    logits = state.apply_fn(
        {"params": params},
        input_tokens,
        deterministic=False,
    ).logits

    loss, accuracy = cross_entropy_loss_and_accuracy(logits, target_tokens, loss_masks)
    return loss, {"accuracy": accuracy}

def train_step(
    state: train_state.TrainState,
    batch: Dict[str, Any],
    rng: jax.random.PRNGKey,
) -> Tuple[train_state.TrainState, jax.random.PRNGKey, Dict[str, Any]]:
    """Performs a single training step."""
    rng_generator = JaxRNG(rng)
    batch = with_sharding_constraint(batch, PS(('dp', 'fsdp')))

    grad_fn = jax.value_and_grad(forward_and_loss_fn, has_aux=True)
    (loss, aux), grads = grad_fn(state.params, state, batch)
    
    state = state.apply_gradients(grads=grads)
    
    metrics = {
        "loss": loss,
        "accuracy": aux["accuracy"],
        "learning_rate": state.opt_state[1].hyperparams["learning_rate"],
        "gradient_norm": global_norm(grads),
        "param_norm": global_norm(state.params),
    }
    
    return state, rng_generator(), metrics


def train_loop(
    model: transformer_lib.Transformer,
    params: Dict[str, Any],
    optimizer: optax.GradientTransformation,
    train_dataloader: torch.utils.data.DataLoader,
    tokenizer: AutoTokenizer,
    training_cfg: Any,
    mesh: Mesh,
) -> train_state.TrainState:
    state_shapes = jax.eval_shape(
        functools.partial(
            create_trainstate_from_params,
            params=params,
            model_apply_fn=model.apply,
            optimizer=optimizer,
        ),
    )
    
    state_shapes_partitioned = shard_params_pytree(state_shapes, mesh)
    
    sharded_train_step = jax.jit(
        train_step,
        in_shardings=(state_shapes_partitioned, NamedSharding(mesh, PS()), NamedSharding(mesh, PS())),
        out_shardings=(state_shapes_partitioned, NamedSharding(mesh, PS()), NamedSharding(mesh, PS())),
    )
    
    state = create_trainstate_from_params(params, model.apply, optimizer)
    state = state.replace(pad_token_id=tokenizer.pad_token_id)
    
    rng = jax.random.PRNGKey(training_cfg.seed)
    
    step_counter = trange(training_cfg.total_steps, ncols=0)
    
    for step in step_counter:
        batch = next(iter(train_dataloader))
        batch = jax.device_put(batch, NamedSharding(mesh, PS()))
        
        state, rng, metrics = sharded_train_step(state, batch, rng)
        
        if step % training_cfg.print_every_n_steps == 0:
            print(f"Step {step}, Loss: {metrics['loss']:.4f}, Accuracy: {metrics['accuracy']:.4f}")
        
        if training_cfg.max_steps and step >= training_cfg.max_steps:
            break
    
    print("Training complete!")
    return state
