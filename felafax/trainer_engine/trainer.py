from dataclasses import dataclass
from typing import Optional, Any
import functools
import pyrallis
import jax
# jax.distributed.initialize()

import equinox as eqx
import quax
import jax.numpy as jnp
from jax.experimental import mesh_utils
import jax.tree_util as jtu
from jax.sharding import NamedSharding, PartitionSpec as PS
from felafax.trainer_engine.utils import named_tree_map

import optax
import os

from felafax.trainer_engine.checkpoint import (
    Checkpointer,
    load_model,
    load_llama_from_hf,
    save_model_to_hf,
)
from felafax.trainer_engine.models.llama3.jax.model import (
    LlamaForCausalLM,
    LlamaConfig,
    LlamaLinear,
)


def get_mesh(num_tpus: int):
    mesh_shape = None
    if num_tpus == 1:
        mesh_shape = (1, 1, 1)
    elif num_tpus == 4:
        mesh_shape = (1, 2, 2)
    elif num_tpus == 8:
        mesh_shape = (2, 2, 2)
    else:
        raise ValueError(f"Invalid number of TPUs: {num_tpus}")

    print(f"Creating TPU device mesh with shape {mesh_shape}...")
    device_mesh = mesh_utils.create_device_mesh(mesh_shape)
    mesh = jax.sharding.Mesh(device_mesh, axis_names=("batch", "fsdp", "mp"))
    return mesh


def merge_lora_params(model):
    def merge_fn(module):
        if (
            isinstance(module, LlamaLinear)
            and module.lora_A is not None
            and module.lora_B is not None
        ):
            delta_weight = module.lora_A @ module.lora_B.T
            new_weight = module.weight + delta_weight
            module = eqx.tree_at(lambda m: m.weight, module, new_weight)

            # Optionally set lora_A and lora_B to None
            module = eqx.tree_at(
                lambda m: (m.lora_A, m.lora_B), module, (None, None)
            )
        return module

    model = jtu.tree_map(merge_fn, model, is_leaf=eqx.is_array)
    return model


# Define configuration flags and default values
@dataclass
class TrainerConfig:
    """Configuration for the Llama trainer"""

    # Model configuration
    model_name: str = "meta-llama/Llama-3.2-1B"
    param_dtype: str = "float32"
    output_dtype: str = "float32"

    # Training configuration
    num_epochs: int = 1
    num_steps: int = 5
    num_tpus: int = jax.device_count()

    # LoRA configuration
    use_lora: bool = True  # Enable or disable LoRA training
    lora_rank: int = 4  # Rank for LoRA matrices

    # Environment configuration
    base_dir: str = "/mnt/persistent-disk"
    hf_token: Optional[str] = None


# Core trainer class -- add non-essential things in private functions.
class Trainer:
    def __init__(
        self,
        trainer_config: TrainerConfig,
        train_dataloader: Any,
        val_dataloader: Any,
        mesh: Optional[jax.sharding.Mesh] = None,
        checkpointer: Optional[Checkpointer] = None,
    ):
        assert trainer_config.model_name, "model_name must be provided"

        self.trainer_config = trainer_config
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.mesh = mesh if mesh else get_mesh(trainer_config.num_tpus)
        self.checkpointer = checkpointer

        # Load the model and model_config
        self.model, self.model_config = load_llama_from_hf(
            model_name=trainer_config.model_name,
            token=trainer_config.hf_token,
            lora_rank=trainer_config.lora_rank if trainer_config.use_lora else 0,
        )

        lora_params, _ = eqx.partition(
            self.model,
            filter_spec=_is_lora_param_filter_spec(self.model),
            is_leaf=eqx.is_array,
        )
        self.configure_optimizers(lora_params)

    def configure_optimizers(self, params):
        self.optimizer = optax.adam(learning_rate=1e-3)
        self.opt_state = self.optimizer.init(params)

    @functools.partial(jax.jit, static_argnames="self")
    def forward(self, lora_params, model_static, batch):
        model = eqx.combine(lora_params, model_static)
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        attention_mask = batch.get("attention_mask", None)
        position_ids = batch.get("position_ids", None)

        logits = model(input_ids, attention_mask, position_ids)

        # Shift for next-token prediction
        shifted_logits = logits[..., :-1, :]  # Remove last logit
        shifted_labels = labels[..., 1:]  # Remove first token

        # If using attention mask, shift it too
        shifted_mask = None
        if attention_mask is not None:
            shifted_mask = attention_mask[..., 1:]

        loss, accuracy = _cross_entropy_loss_and_accuracy(
            shifted_logits, shifted_labels, shifted_mask
        )
        return loss, accuracy

    @functools.partial(jax.jit, static_argnames=("self", "optimizer"))
    def training_step(
        self, lora_params, model_static, optimizer, optimizer_state, batch
    ):
        grad_fn = jax.value_and_grad(self.forward, argnums=0, has_aux=True)
        (loss, accuracy), grads = grad_fn(
            lora_params,
            model_static=model_static,
            batch=batch,
        )

        updates, optimizer_state = optimizer.update(
            grads, optimizer_state, lora_params
        )
        lora_params = optax.apply_updates(lora_params, updates)

        return loss, (accuracy, lora_params, optimizer_state)

    def validation_step(
        self, *, model_params, model_static, optimizer, optimizer_state, batch
    ):
        pass

    def train(self):
        lora_params, model_static = eqx.partition(
            self.model,
            filter_spec=_is_lora_param_filter_spec(self.model),
            is_leaf=eqx.is_inexact_array,
        )
        optimizer_state = self.opt_state
        max_steps = self.trainer_config.num_steps or float("inf")

        prev_step = 0
        prev_loss = 0.0
        prev_accuracy = 0.0

        for step, batch in enumerate(self.train_dataloader):
            if step >= max_steps:
                break

            if step:
                # Printing metrics of previous step, so that XLA pipelining is not disrupted.
                print(
                    f"Step {prev_step}: Loss: {prev_loss:.4f}, Accuracy: {prev_accuracy:.4f}"
                )

            batch = _preprocess_batch(batch)
            batch = jax.device_put(batch, NamedSharding(self.mesh, PS("batch")))
            optimizer_state = jax.device_put(
                optimizer_state, NamedSharding(self.mesh, PS())
            )

            loss, (accuracy, lora_params, optimizer_state) = self.training_step(
                lora_params=lora_params,
                model_static=model_static,
                optimizer=self.optimizer,
                optimizer_state=optimizer_state,
                batch=batch,
            )

            prev_step = step + 1
            prev_loss = loss
            prev_accuracy = accuracy

            if self.checkpointer:
                self.checkpointer.save_checkpoint(
                    model=eqx.combine(lora_params, model_static),
                    model_config=self.model_config,
                    step=step + 1,
                )

        # Save final checkpoint
        if self.checkpointer:
            self.checkpointer.save_checkpoint(
                model=eqx.combine(lora_params, model_static),
                model_config=self.model_config,
                step=step + 1,
            )
            self.checkpointer.wait_until_finished()
            print("Final checkpoint saved at:", self.checkpointer.directory)

        self.model = eqx.combine(lora_params, model_static)
        print("Training completed!")

        # Call export method to save the model
        self.export()

    def export(self):
        # After training, convert and save the model in Hugging Face format
        self.model = merge_lora_params(self.model)
        export_dir = os.path.join(self.trainer_config.base_dir, "hf_export")
        save_model_to_hf(
            model=self.model,
            model_config=self.model_config,
            output_dir=export_dir,
            tokenizer_name=self.trainer_config.model_name,
        )
        print("Hugging Face model saved at:", export_dir)


def _preprocess_batch(batch):
    # Convert PyTorch tensors to JAX arrays
    batch = {
        k: v if isinstance(v, jax.Array) else jax.numpy.array(v.numpy())
        for k, v in batch.items()
    }
    batch["input_ids"] = batch["input_ids"].astype(jnp.int32)
    batch["labels"] = batch["labels"].astype(jnp.int32)

    # Add position IDs to batch
    seq_length = batch["input_ids"].shape[1]
    batch["position_ids"] = jnp.repeat(
        jnp.arange(seq_length)[None, :],
        batch["input_ids"].shape[0],
        axis=0,
    )
    return batch


def _is_lora_param_filter_spec(model):
    is_lora_param_filter_spec = named_tree_map(
        lambda path_str, value: "lora_A" in path_str or "lora_B" in path_str,
        model,
        is_leaf=eqx.is_array,
    )
    return is_lora_param_filter_spec


def _cross_entropy_loss_and_accuracy(logits, tokens, mask=None):
    if mask is None:
        mask = jnp.ones(tokens.shape[:2])
    mask = mask.astype(jnp.float32)

    valid_text_length = jnp.maximum(jnp.sum(mask, axis=-1), 1e-10)

    logits = logits.astype(jnp.float32)  # for numerical stability
    token_log_prob = jnp.squeeze(
        jnp.take_along_axis(
            jax.nn.log_softmax(logits, axis=-1),
            jnp.expand_dims(tokens, -1),
            axis=-1,
        ),
        -1,
    )
    token_log_prob = jnp.where(mask > 0.0, token_log_prob, jnp.array(0.0))
    loss = -jnp.mean(jnp.sum(token_log_prob, axis=-1) / valid_text_length)
    correct = jnp.where(
        mask > 0.0, jnp.argmax(logits, axis=-1) == tokens, jnp.array(False)
    )
    accuracy = jnp.mean(jnp.sum(correct, axis=-1) / valid_text_length)
    return loss, accuracy
