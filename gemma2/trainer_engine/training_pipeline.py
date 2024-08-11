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


def forward_and_loss_fn(
    params,
    state,
    input_tokens: jax.Array,  # Shape [B, L]
    input_mask: jax.Array,  # Shape [B, L]
    positions: jax.Array,  # Shape [B, L]
    attention_mask: jax.Array,  # [B, L, L]
) -> jax.Array:
    """Forward pass and loss function.

    Args:
      params: model's input parameters.
      model: gemma transformer model to call.
      input_tokens: input tokens sequence, shape [B, L].
      input_mask: tokens to ignore when computing the loss, shape [B, L].
      positions: relative position of each token, shape [B, L].
      attention_mask: input attention mask, shape [B, L].

    Returns:
      Softmax cross-entropy loss for the next-token prediction task.
    """

    # Forward pass on the input data.
    # No attention cache is needed here.
    logits, _ = state.apply_fn(
        {"params": params},
        input_tokens,
        positions,
        None,  # Attention cache is None.
        attention_mask,
    )

    # Exclude the last step as it does not appear in the targets.
    logits = logits[:, :-1]

    # Similarly, the first token cannot be predicteds.
    target_tokens = input_tokens[:, 1:]
    target_mask = input_mask[:, 1:]

    # Convert the target labels into one-hot encoded vectors.
    one_hot = jax.nn.one_hot(target_tokens, logits.shape[-1])

    # Don't update on unwanted tokens.
    one_hot = one_hot * target_mask.astype(one_hot.dtype)[..., None]

    # Normalisation factor.
    norm_factor = 1 / (jnp.sum(target_mask) + 1e-8)

    # Return the nll loss.
    loss = -jnp.sum(jax.nn.log_softmax(logits) * one_hot) * norm_factor
    return loss


def get_attention_mask_and_positions(
    token_ids: jax.Array, padding_token_id: int
) -> tuple[jax.Array, jax.Array]:
    """
    Creates attention mask and position encodings for transformer input.

    This function prepares two essential components for transformer processing:
    1. Attention mask: Ensures the model attends only to relevant tokens.
    2. Position encodings: Assign a unique position to each non-padding token.

    Args:
        token_ids (jax.Array): A 2D array of token IDs, where each row represents
                               a sequence in the batch.
        padding_token_id (int): The ID used for padding tokens.

    Returns:
        tuple[jax.Array, jax.Array]: A tuple containing:
            - position_encodings: A 2D array where each non-padding token is
              assigned a unique position, starting from 0.
            - attention_mask: A 3D array representing the causal attention mask,
              where 1 indicates allowed attention and 0 indicates masked attention.

    Note:
        - Padding tokens are used to make all sequences in a batch the same length.
        - The padding mask identifies actual input tokens (True) vs padding tokens (False).
        - Position encodings help the model understand the order of tokens in the sequence.
        - The attention mask ensures the model doesn't attend to or learn from padding tokens
          and implements causal attention (each token can only attend to itself and preceding tokens).
    """
    # Create padding mask: True for real tokens, False for padding
    padding_mask = token_ids != padding_token_id

    # Generate position encodings based on the padding mask
    position_encodings = transformer_lib.build_positions_from_mask(padding_mask)

    # Create causal attention mask
    attention_mask = transformer_lib.make_causal_attn_mask(padding_mask)

    return position_encodings, attention_mask


def train_step(
    state: train_state.TrainState, batch: Dict[str, Any], padding_token_id: int
) -> Tuple[Any, float]:
    """Performs a single training step.

    Args:
      model: gemma transformer model.
      params: model's input parameters.
      pad_id: id of the pad token.
      batch: input batch.

    Returns:
      Training loss, updated parameters, updated optimizer state.
    """
    # Build the position and attention mask vectors.
    positions, attention_mask = get_attention_mask_and_positions(
        batch["input_tokens"], padding_token_id
    )

    # Forward and backward passes
    train_loss, grads = jax.value_and_grad(forward_and_loss_fn)(
        state.params,
        state=state,
        input_tokens=batch["input_tokens"],
        input_mask=batch["target_mask"],
        positions=positions,
        attention_mask=attention_mask,
    )

    # Update the parameters
    state = state.apply_gradients(grads=grads)

    return state, train_loss


def shard_params_pytree(params, mesh):
    def shard_param(param):
        if len(param.shape) == 0:
            return NamedSharding(mesh, PS())
        elif len(param.shape) == 1:
            return NamedSharding(mesh, PS("model"))
        elif len(param.shape) == 2:
            return NamedSharding(mesh, PS("data", "model"))
        elif len(param.shape) == 3:
            return NamedSharding(mesh, PS("data", "model", "replica"))
        else:
            # For higher-dimensional tensors, might need a more complex strategy. But reeplicate by default fornow.
            return NamedSharding(mesh, PS())

    return jax.tree_util.tree_map(shard_param, params)


def create_trainstate_from_params(params, model_apply_fn, optimizer):
    state = train_state.TrainState.create(
        apply_fn=model_apply_fn, params=params["params"], tx=optimizer
    )
    return state


def train_loop(
    model: transformer_lib.Transformer,
    params: Dict[str, Any],
    optimizer: optax.GradientTransformation,
    train_dataloader: torch.utils.data.DataLoader,
    tokenizer: AutoTokenizer,
    training_cfg: Any,
    mesh: Mesh,
) -> train_state.TrainState:
    """
    Execute the training loop for the Gemma model.

    This function performs the following steps:
    1. Evaluate the shape of the train state
    2. Shard the parameters across devices
    3. Create a JIT-compiled, sharded training step
    4. Iterate through the training data, updating the model state
    """

    # Evaluate the shape of the train state
    state_shapes = jax.eval_shape(
        functools.partial(
            create_trainstate_from_params,
            params=params,
            model_apply_fn=model.apply,
            optimizer=optimizer,
        ),
    )

    # Shard the parameters across devices
    state_shapes_partitioned = shard_params_pytree(state_shapes, mesh)

    # Create a JIT-compiled, sharded training step
    sharded_train_step = jax.jit(
        train_step,
        in_shardings=(state_shapes_partitioned, NamedSharding(mesh, PS())),
        out_shardings=(state_shapes_partitioned, NamedSharding(mesh, PS())),
        static_argnums=(2,),
    )

    n_steps: int = 0
    avg_loss: float = 0.0

    # Initialize the training state with unsharded parameters
    state = create_trainstate_from_params(params, model.apply, optimizer)

    # Iterate through the training data
    for i, train_batch in enumerate(train_dataloader):
        # Place the batch on the appropriate devices
        train_batch = jax.device_put(train_batch, NamedSharding(mesh, PS()))

        # Perform a single training step
        state, train_loss = sharded_train_step(
            state, train_batch, tokenizer.pad_token_id
        )

        n_steps += 1
        avg_loss += train_loss
        print(f"Step {n_steps}, Train Loss: {train_loss:.4f}")

        if training_cfg.max_steps is not None and n_steps >= training_cfg.max_steps:
            break

    avg_loss /= n_steps
    print(f"Training complete. Average loss: {avg_loss:.4f}")

    return state
