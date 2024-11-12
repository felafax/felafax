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
from jax.sharding import NamedSharding, PartitionSpec as PS

import optax

from felafax.trainer_engine.checkpoint import (
    Checkpointer,
    load_model,
    load_checkpoint_or_model,
)
from felafax.trainer_engine.data.alpaca import AlpacaDataset
from transformers import AutoTokenizer


def get_mesh(num_tpus: int):
    mesh_shape = None
    if num_tpus == 4:
        mesh_shape = (1, 2, 2)
    elif num_tpus == 8:
        mesh_shape = (2, 2, 2)
    else:
        raise ValueError(f"Invalid number of TPUs: {num_tpus}")

    print(f"Creating TPU device mesh with shape {mesh_shape}...")
    device_mesh = mesh_utils.create_device_mesh(mesh_shape)
    mesh = jax.sharding.Mesh(
        device_mesh, axis_names=("batch", "fsdp", "replica")
    )
    return mesh


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


# Define configuration flags and default values
@dataclass
class TrainerConfig:
    """Configuration for the Llama trainer"""

    # Model configuration
    model_name: str = "meta-llama/Llama-3.2-1B"
    checkpoint_dir: str = "/mnt/persistent-disk/models/llama3.2-1b/"
    param_dtype: str = "float32"
    output_dtype: str = "float32"

    # Training configuration
    num_epochs: int = 1
    num_steps: int = 5
    batch_size: int = 8
    seq_length: int = 512

    # Hardware/parallelism configuration
    num_tpus: int = jax.device_count()
    num_dataloader_workers: int = 4


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

        self.model, self.model_config = load_model(
            model_name=trainer_config.model_name,
        )
        self.model = quax.lora.loraify(
            self.model, rank=2, key=jax.random.PRNGKey(0)
        )
        self.configure_optimizers()

    def configure_optimizers(self):
        self.optimizer = optax.adam(learning_rate=1e-3)
        self.opt_state = self.optimizer.init(
            eqx.filter(self.model, eqx.is_array)
        )

    # TODO: Add microbatching (nando ref).
    @functools.partial(
        jax.jit, static_argnames=("self", "model_static", "optimizer")
    )
    def forward(
        self, model_params, model_static, optimizer, optimizer_state, batch
    ):
        """Computes loss for a single forward and backward pass."""
        model = eqx.combine(model_params, model_static)

        input_ids = batch["input_ids"]
        input_ids = input_ids.astype(jnp.int32)
        attention_mask = batch.get("attention_mask", None)
        position_ids = batch.get("position_ids", None)

        logits = model(input_ids, attention_mask, position_ids)
        loss, accuracy = _cross_entropy_loss_and_accuracy(logits, input_ids)
        return loss, (accuracy, model, optimizer_state)

    def training_step(
        self, model_params, model_static, optimizer, optimizer_state, batch
    ):
        grad_fn = jax.value_and_grad(self.forward, argnums=(0), has_aux=True)
        (loss, (accuracy, model, optimizer_state)), grads = grad_fn(
            model_params,
            model_static=model_static,
            optimizer=optimizer,
            optimizer_state=optimizer_state,
            batch=batch,
        )

        updates, optimizer_state = optimizer.update(
            grads, optimizer_state, model_params
        )
        model_params = optax.apply_updates(model_params, updates)
        return loss, (accuracy, model_params, optimizer_state)

    def validation_step(
        self, *, model_params, model_static, optimizer, optimizer_state, batch
    ):
        pass

    def train(self):
        model_params, model_static = eqx.partition(self.model, eqx.is_array)
        optimizer_state = self.opt_state
        max_steps = self.trainer_config.num_steps or float("inf")

        prev_step = 0
        prev_loss = 0.0
        prev_accuracy = 0.0

        for step, batch in enumerate(self.train_dataloader):
            if step >= max_steps:
                break

            batch = _preprocess_batch(batch)
            batch = jax.device_put(batch, NamedSharding(self.mesh, PS("batch")))
            optimizer_state = jax.device_put(
                optimizer_state, NamedSharding(self.mesh, PS())
            )

            loss, (accuracy, model_params, optimizer_state) = self.training_step(
                model_params=model_params,
                model_static=model_static,
                optimizer=self.optimizer,
                optimizer_state=optimizer_state,
                batch=batch,
            )

            print(
                f"Step {prev_step}: Loss: {prev_loss:.4f}, Accuracy: {prev_accuracy:.4f}"
            )
            prev_step = step + 1
            prev_loss = loss
            prev_accuracy = accuracy

            if self.checkpointer:
                self.checkpointer.save_checkpoint(
                    model=eqx.combine(model_params, model_static),
                    model_config=self.model_config,
                    step=step + 1,
                )

        # Save final checkpoint
        if self.checkpointer:
            self.checkpointer.save_checkpoint(
                model=eqx.combine(model_params, model_static),
                model_config=self.model_config,
                step=step + 1,
            )
            self.checkpointer.wait_until_finished()
            print("Final checkpoint saved at:", self.checkpointer.checkpoint_dir)

            # Load checkpoint to test
            model, model_config = load_checkpoint_or_model(
                model_name=self.trainer_config.model_name,
                checkpointer=self.checkpointer,
            )
            print("Model was restored!")

        self.model = eqx.combine(model_params, model_static)
        print("Training completed!")


if __name__ == "__main__":
    trainer_config = pyrallis.parse(config_class=TrainerConfig)

    # Set up tokenizer
    tokenizer = AutoTokenizer.from_pretrained(trainer_config.model_name)

    # Initialize the Alpaca dataset
    data_module = AlpacaDataset(
        batch_size=trainer_config.batch_size,
        max_seq_length=trainer_config.seq_length,
        num_workers=trainer_config.num_dataloader_workers,
    )
    data_module.setup(tokenizer=tokenizer)
    train_dataloader = data_module.train_dataloader()
    val_dataloader = data_module.val_dataloader()

    # Initialize Checkpointer
    checkpointer = Checkpointer(trainer_config.checkpoint_dir)

    # Initialize Trainer with Checkpointer
    trainer = Trainer(
        trainer_config=trainer_config,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        checkpointer=checkpointer,
    )
    trainer.train()
