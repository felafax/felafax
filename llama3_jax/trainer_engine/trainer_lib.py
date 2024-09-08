import functools
import gc
import json
import os
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import pdb

import chex
import flax
import jax
import jax.numpy as jnp
import torch
from flax.training import train_state
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as PS
from transformers import LlamaConfig, LlamaForCausalLM

from . import checkpoint_lib, jax_utils, utils
from .jax_utils import cross_entropy_loss_and_accuracy


class FelafaxTrainer(ABC):

    @abstractmethod
    def setup(self):
        pass

    @abstractmethod
    def train_step(self, state, batch, rng):
        pass

    @abstractmethod
    def eval_step(self, state, batch):
        pass

    @abstractmethod
    def save_checkpoint(self, state, path):
        pass

    @abstractmethod
    def load_checkpoint(self, path):
        pass

    @abstractmethod
    def compute_loss(self, logits, labels, mask):
        pass


class CausalLMTrainer(FelafaxTrainer):

    def __init__(
            self,
            model,
            model_ckpt_path,
            model_configurator,
            optimizer,
            training_config,
            mesh,
            model_name: str = None,
            model_params: Dict[str, Any] = None,
            compiled_train_step_path: str = None,  # New parameter
    ):
        self.model = model
        self.model_ckpt_path = model_ckpt_path
        self.model_configurator = model_configurator
        self.optimizer = optimizer
        self.training_config = training_config
        self.mesh = mesh
        self.model_name = model_name
        self.model_params = model_params
        self.compiled_train_step_path = compiled_train_step_path or "/mnt/persistent-disk/compiled/compiled_train_step.pkl"

        self.compiled_train_step = None
        self.setup()

    def setup(self):
        self.checkpointer = checkpoint_lib.Checkpointer(
            checkpoint_lib.Checkpointer.get_default_config(),
            checkpoint_dir=os.path.dirname(self.model_ckpt_path),
            enable_checkpointer=jax.process_index() == 0,
        )

        self.state_shapes = self.get_state_shapes()
        self.state_shapes_partitioned = jax_utils.match_partition_rules(
            self.model_configurator.get_partition_rules(), self.state_shapes)

        self.shard_fns, self.gather_fns = checkpoint_lib.make_shard_and_gather_fns(
            self.state_shapes_partitioned, self.state_shapes)

        jax_utils.init_rng(99)
        jax_utils.next_rng()

        with self.mesh:
            print("Loading causal language model...")
            if self.model_params is None:
                if self.model_name is None or self.model_name in [
                        "llama-3.1-8B-Instruct-JAX", "llama-3.1-8B-JAX"
                ]:
                    _, self.model_params = (
                        self.checkpointer.load_trainstate_checkpoint(
                            "flax_params::" + self.model_ckpt_path,
                            self.state_shapes, self.shard_fns))
                else:
                    _, self.model_params = (
                        self.checkpointer.load_trainstate_checkpoint(
                            "params::" + self.model_ckpt_path,
                            self.state_shapes, self.shard_fns))

            if self.model_params is not None:
                self.train_state = self.create_train_state_from_params(
                    self.model_params)
            else:
                raise ValueError("Failed to load checkpoint")

        # self.load_or_compile_train_step()

    def load_or_compile_train_step(self):
        try:
            self.compiled_train_step = utils.load_pickle(
                self.compiled_train_step_path)
            print(
                f"Loaded compiled train step from {self.compiled_train_step_path}"
            )
        except FileNotFoundError:
            print(
                f"Compiled train step not found at {self.compiled_train_step_path}. Compiling now..."
            )
            self.compile_train_step()

    def compile_train_step(self):
        print("Compiling train step...")
        dummy_state = self.create_train_state_from_params(self.model_params)
        dummy_batch = self.get_dummy_batch()

        jitted_train_step = jax.jit(
            self.train_step,
            in_shardings=(
                self.state_shapes_partitioned,  # state
                NamedSharding(self.mesh, PS("dp")),  # batch
                NamedSharding(self.mesh, PS()),  # rng
            ),
            out_shardings=(
                self.state_shapes_partitioned,  # updated state
                NamedSharding(self.mesh, PS()),  # new rng
                NamedSharding(self.mesh, PS())  # metrics
            ))

        # AOT
        self.compiled_train_step = jitted_train_step.lower(
            dummy_state, dummy_batch, jax.random.PRNGKey(0)).compile()

        # Save the compiled function
        utils.save_pickle(self.compiled_train_step,
                          self.compiled_train_step_path)
        print(f"Compiled train step saved to {self.compiled_train_step_path}")

    def get_dummy_batch(self):
        # Create a dummy batch matching your expected input structure
        return {
            "input_tokens":
            jnp.zeros((4, self.training_config.seq_length), dtype=jnp.int32),
            "target_tokens":
            jnp.zeros((4, self.training_config.seq_length), dtype=jnp.int32),
            "loss_masks":
            jnp.ones((4, self.training_config.seq_length), dtype=jnp.float32),
        }

    def get_state_shapes(self):
        return jax.eval_shape(
            functools.partial(
                self.initialize_state,
                rng=jax.random.PRNGKey(0),
                model=self.model,
                model_config=self.model_configurator,
                seq_length=self.training_config.seq_length,
                optimizer=self.optimizer,
            ))

    @staticmethod
    def initialize_state(rng, model, model_config, seq_length, optimizer):
        rng_generator = jax_utils.NextRNG(rng)

        params = model.init(
            input_ids=jnp.zeros((4, seq_length), dtype=jnp.int32),
            position_ids=jnp.zeros((4, seq_length), dtype=jnp.int32),
            attention_mask=jnp.ones((4, seq_length), dtype=jnp.int32),
            rngs=rng_generator(model_config.get_rng_keys()),
        )
        return train_state.TrainState.create(params=params,
                                             tx=optimizer,
                                             apply_fn=model.apply)

    def create_train_state_from_params(self, params):
        return train_state.TrainState.create(params=params,
                                             apply_fn=self.model.apply,
                                             tx=self.optimizer)

    @property
    def jitted_train_step(self):
        return jax.jit(
            self.train_step,
            in_shardings=(
                self.state_shapes_partitioned,  # state
                NamedSharding(self.mesh, PS("dp")),  # batch
                NamedSharding(self.mesh, PS()),  # rng
            ),
            out_shardings=(
                self.state_shapes_partitioned,  # updated state
                NamedSharding(self.mesh, PS()),  # new rng
                NamedSharding(self.mesh, PS())  # metrics
            ))

    def train_step(self, state, batch, rng):
        rng_generator = jax_utils.NextRNG(rng)

        def loss_and_accuracy(params):
            # Reshape the input tensors to combine the data parallel dimension with the batch dimension
            input_tokens = batch["input_tokens"].reshape(
                -1, batch["input_tokens"].shape[-1])
            target_tokens = batch["target_tokens"].reshape(
                -1, batch["target_tokens"].shape[-1])
            loss_masks = batch["loss_masks"].reshape(
                -1, batch["loss_masks"].shape[-1])

            logits = state.apply_fn(
                params,
                input_tokens,
                deterministic=False,
                rngs=rng_generator(('params', 'dropout', 'fcm')),
            ).logits
            return self.compute_loss(logits, target_tokens, loss_masks)

        grad_fn = jax.value_and_grad(loss_and_accuracy, has_aux=True)
        (loss, accuracy), grads = grad_fn(state.params)

        # Gather gradients from all devices
        (loss, accuracy), grads = grad_fn(state.params)

        state = state.apply_gradients(grads=grads)
        metrics = dict(
            loss=loss,
            accuracy=accuracy,
        )
        return state, rng_generator(), metrics

    @property
    def jitted_eval_step(self):
        return jax.jit(
            self.eval_step,
            in_shardings=(
                self.state_shapes_partitioned,  # state
                NamedSharding(self.mesh, PS("dp")),  # batch
                NamedSharding(self.mesh, PS()),  # rng
            ),
            out_shardings=NamedSharding(self.mesh, PS())  # metrics
        )

    def eval_step(self, state, batch):
        logits = state.apply_fn(
            state.params,
            batch["input_tokens"],
            deterministic=True,
        ).logits

        loss, accuracy = self.compute_loss(logits, batch["target_tokens"],
                                           batch["loss_masks"])
        metrics = dict(
            loss=loss,
            accuracy=accuracy,
        )
        return metrics

    def train(self,
              train_dataloader,
              eval_dataloader,
              run_jitted=True,
              run_aot=False):
        state = self.train_state

        for epoch in range(self.training_config.num_epochs):
            print(f"Starting epoch {epoch} of training...")

            for step, train_batch in enumerate(train_dataloader):
                # Reshape the batch for data parallelism
                train_batch = jax.device_put(train_batch, NamedSharding(self.mesh, PS("dp", None)))

                sharded_rng = jax_utils.next_rng()

                if run_aot and self.compiled_train_step is not None:
                    state, sharded_rng, metrics = self.compiled_train_step(
                        state, train_batch, sharded_rng)
                elif run_jitted:
                    state, sharded_rng, metrics = self.jitted_train_step(
                        state, train_batch, sharded_rng)
                else:
                    state, sharded_rng, metrics = self.train_step(
                        state, train_batch, sharded_rng)

                if step % self.training_config.print_every_n_steps == 0:
                    print(
                        f"Epoch {epoch}, Step {step}, Train Loss: {metrics['loss']:.4f}, Accuracy: {metrics['accuracy']:.4f}"
                    )

                # if step % self.training_config.eval_every_n_steps == 0:
                #     eval_metrics = self.evaluate(state, eval_dataloader)
                #     print(
                #         f"Epoch {epoch}, Step {step}, Eval Loss: {eval_metrics['loss']:.4f}, Accuracy: {eval_metrics['accuracy']:.4f}"
                #     )

                if (self.training_config.max_steps
                        and step >= self.training_config.max_steps):
                    break

        self.train_state = state
        return state

    def evaluate(self, state, eval_dataloader, run_jitted=True):
        total_loss = 0
        total_accuracy = 0
        num_batches = 0

        for step, eval_batch in enumerate(eval_dataloader):
            if self.training_config.max_eval_steps and step >= self.training_config.max_eval_steps:
                break

            eval_batch = jax.device_put(eval_batch,
                                        NamedSharding(self.mesh, PS()))

            if run_jitted:
                metrics = self.jitted_eval_step(state, eval_batch)
            else:
                metrics = self.eval_step(state, eval_batch)

            total_loss += metrics['loss']
            total_accuracy += metrics['accuracy']
            num_batches += 1

            print(
                f"Eval Step {step}, Loss: {metrics['loss']:.4f}, Accuracy: {metrics['accuracy']:.4f}"
            )

        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches
        print(
            f"Evaluation complete. Average Loss: {avg_loss:.4f}, Average Accuracy: {avg_accuracy:.4f}"
        )

        return {'loss': avg_loss, 'accuracy': avg_accuracy}

    def save_checkpoint(self, state, path):
        print(f"Saving checkpoint to {path}...")
        self.checkpointer.save_checkpoint_simple(train_state=state,
                                                 filename=path)
        print(f"Checkpoint saved to {path}.")

    def load_checkpoint(self, path):
        _, params = self.checkpointer.load_trainstate_checkpoint(
            path, self.state_shapes, self.shard_fns)
        return self.create_train_state_from_params(params)

    def compute_loss(self, logits, labels, mask):
        return cross_entropy_loss_and_accuracy(logits, labels, mask)
