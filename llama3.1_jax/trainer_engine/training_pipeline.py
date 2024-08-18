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

# Third-party imports
import numpy as np
from tqdm import tqdm, trange
from transformers import AutoTokenizer

# JAX and related libraries
import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
from flax.training import train_state
from flax.core.meta import unbox
from flax.serialization import from_bytes, to_bytes, to_state_dict, from_state_dict
from flax.training.train_state import TrainState
import optax

# JAX model partitioning and sharding
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as PS
from jax.experimental import mesh_utils
from jax.experimental.pjit import pjit

# Local imports
from trainer_engine import jax_utils
from trainer_engine.jax_utils import cross_entropy_loss_and_accuracy
from trainer_engine import llama_model
from trainer_engine import checkpointer

# Less common imports
import mlxu
import torch


def init_fn(rng, model, seq_length, optimizer):
    rng_generator = jax_utils.JaxRNG(rng)
    params = model.init(
        input_ids=jnp.zeros((4, seq_length), dtype=jnp.int32),
        position_ids=jnp.zeros((4, seq_length), dtype=jnp.int32),
        attention_mask=jnp.ones((4, seq_length), dtype=jnp.int32),
        rngs=rng_generator(llama_model.LlamaConfig.rng_keys()),
    )
    return TrainState.create(params=params, tx=optimizer, apply_fn=model.apply)


def get_state_shapes(model, seq_length, optimizer):
    return jax.eval_shape(
        functools.partial(
            init_fn,
            rng=jax.random.PRNGKey(0),
            model=model,
            seq_length=seq_length,
            optimizer=optimizer,
        )
    )


def create_trainstate_from_params(params, model_apply_fn, optimizer):
    return TrainState.create(params=params, apply_fn=model_apply_fn, tx=optimizer)


def get_sharded_create_trainstate_from_params(state_partitioned):
    return pjit(
        create_trainstate_from_params,
        in_shardings=(state_partitioned.params,),
        out_shardings=state_partitioned,
        static_argnums=(1, 2),
        # donate_argnums=(0, ),
    )


def train_step(state, rng, batch):
    rng_generator = jax_utils.JaxRNG(rng)
    batch = jax_utils.with_sharding_constraint(batch, PS(("dp", "fsdp")))

    def loss_and_accuracy(params):
        logits = state.apply_fn(
            params,
            batch["input_tokens"],
            deterministic=False,
            rngs=rng_generator(llama_model.LlamaConfig.rng_keys()),
        ).logits
        return jax_utils.cross_entropy_loss_and_accuracy(
            logits, batch["target_tokens"], batch["loss_masks"]
        )

    grad_fn = jax.value_and_grad(loss_and_accuracy, has_aux=True)
    (loss, accuracy), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    metrics = dict(
        loss=loss,
        accuracy=accuracy,
    )
    return state, rng_generator(), metrics


def get_sharded_train_step(state_partitioned):
    return pjit(
        functools.partial(train_step),
        in_shardings=(state_partitioned, PS(), PS()),
        out_shardings=(state_partitioned, PS(), PS()),
        donate_argnums=(0, 1),
    )


def train_loop(
    model: Any,
    optimizer: optax.GradientTransformation,
    train_dataloader: torch.utils.data.DataLoader,
    tokenizer: AutoTokenizer,
    training_cfg: Any,
    mesh: Mesh,
    model_path: str,
    params: Dict[str, Any],
) -> train_state.TrainState:
    # initalizes rng generator in jax_utils
    jax_utils.init_rng(99)
    jax_utils.next_rng()

    devices = jax.devices()
    device_count = len(devices)
    device_mesh = mesh_utils.create_device_mesh((1, device_count, 1))
    mesh = Mesh(devices=device_mesh, axis_names=("dp", "fsdp", "mp"))

    state_shapes = get_state_shapes(model, training_cfg.seq_length, optimizer)

    state_shapes_partitioned = jax_utils.match_partition_rules(
        llama_model.LlamaConfig.get_partition_rules(), state_shapes
    )

    shard_fns, gather_fns = jax_utils.make_shard_and_gather_fns(
        state_shapes_partitioned, state_shapes
    )

    sharded_train_step = get_sharded_train_step(state_shapes_partitioned)
    sharded_create_trainstate_from_params = get_sharded_create_trainstate_from_params(
        state_shapes_partitioned
    )

    streaming_checkpointer = checkpointer.StreamingCheckpointer(
        checkpointer.StreamingCheckpointer.get_default_config(),
        checkpoint_dir=os.path.dirname(model_path),
        enable=jax.process_index() == 0,
    )

    with mesh:
        state, restored_params = None, None
        state, restored_params = streaming_checkpointer.load_trainstate_checkpoint(
            "flax_params::" + model_path, state_shapes, shard_fns
        )
        if restored_params is not None:
            state = sharded_create_trainstate_from_params(
                restored_params, model.apply, optimizer
            )
            del restored_params
        else:
            raise ValueError("Failed to load checkpoint")

        for epoch in range(training_cfg.num_epochs):
            for step, train_batch in enumerate(train_dataloader):
                # Place the batch on the appropriate devices
                train_batch = jax.device_put(train_batch, NamedSharding(mesh, PS()))

                sharded_rng = jax_utils.next_rng()

                # Perform a single training step
                state, sharded_rng, metrics = sharded_train_step(
                    state, sharded_rng, train_batch
                )

                if step % training_cfg.print_every_n_steps == 0:
                    print(
                        f"Epoch {epoch}, Step {step}, Train Loss: {metrics['loss']:.4f}, Accuracy: {metrics['accuracy']:.4f}"
                    )

                if training_cfg.max_steps and step >= training_cfg.max_steps:
                    break
        return state
