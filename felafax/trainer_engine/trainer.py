from dataclasses import dataclass
from typing import Optional
import pyrallis
import jax
# jax.distributed.initialize()

import equinox as eqx
import jax.numpy as jnp
from jax.experimental import mesh_utils
from jax.sharding import NamedSharding, PartitionSpec as PS

import optax

from felafax.trainer_engine.checkpoint import load_checkpoint

# I've looked at maxtext code -- not having class makes things super complex. You literally have to written some 10 things frm some funcitons instead of updating a class variable.


def _get_dummy_data(trainer_config):
    # TODO: I'll merge with Alpaca dataset pipeline I have, for now just use dummyd data.
    input_ids = jnp.zeros(
        (trainer_config.batch_size, trainer_config.seq_length), dtype=jnp.int32
    )
    attention_mask = jnp.ones(
        (trainer_config.batch_size, trainer_config.seq_length), dtype=jnp.int32
    )
    position_ids = jnp.repeat(
        jnp.arange(0, trainer_config.seq_length)[None, :],
        trainer_config.batch_size,
        axis=0,
    )
    return input_ids, attention_mask, position_ids


def _get_mesh(trainer_config):
    mesh_shape = None
    if trainer_config.num_tpus == 4:
        mesh_shape = (1, 2, 2)
    elif trainer_config.num_tpus == 8:
        mesh_shape = (2, 2, 2)
    else:
        raise ValueError(f"Invalid number of TPUs: {trainer_config.num_tpus}")

    print(f"Creating TPU device mesh with shape {mesh_shape}...")
    device_mesh = mesh_utils.create_device_mesh(mesh_shape)
    mesh = jax.sharding.Mesh(device_mesh, axis_names=("batch", "fsdp", "dp"))
    return mesh


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

    model_path: str = "meta-llama/Llama-3.2-1B"
    seq_length: int = 512
    batch_size: int = 8
    num_steps: int = 10
    param_dtype: str = "float32"
    output_dtype: str = "float32"
    num_tpus: int = 4


# CORE TRAINER CLASS -- you can add less core things in private functions.
class Trainer:
    def __init__(
        self,
        trainer_config: TrainerConfig,
        model: Optional[eqx.Module] = None,
        mesh: Optional[jax.sharding.Mesh] = None,
    ):
        self.trainer_config = trainer_config
        self.mesh = mesh if mesh else _get_mesh(trainer_config)

        # Use provided model or load from checkpoint
        if model is not None:
            self.model = model
        elif trainer_config.model_path is not None:
            self.model = load_checkpoint(trainer_config.model_path)
        else:
            raise ValueError("Either model or model_path must be provided")

        self.configure_optimizers()
        pass

    def configure_optimizers(self):
        self.optimizer = optax.sgd(learning_rate=1e-3)
        self.opt_state = self.optimizer.init(
            eqx.filter(self.model, eqx.is_array)
        )
        pass

    # Don't need separate forward and backward pass. In eval step, I anyways have to call inference on equinox model. So, just combine the two steps. So that you can JIT compute loss at once and this can be later provided within model itself by other models.
    # TODO: need to look into microbatching (nando has it).
    def forward(
        self, model_params, model_static, optimizer, optimizer_state, batch
    ):
        """Computes loss for a single forward and backward pass."""

        model = eqx.combine(model_params, model_static)
        input_ids, attention_mask, position_ids = batch
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
        model = eqx.apply_updates(model, updates)
        return loss, (accuracy, model, optimizer_state)

    def validation_step(
        self, *, model_params, model_static, optimizer, optimizer_state, batch
    ):
        pass

    def train(self):
        batch = _get_dummy_data(self.trainer_config)
        batch_sharded = jax.device_put(
            batch, NamedSharding(self.mesh, PS("batch"))
        )

        model_params, model_static = eqx.partition(self.model, eqx.is_array)

        for i in range(self.trainer_config.num_steps):
            loss, (accuracy, model, optimizer_state) = self.training_step(
                model_params=model_params,
                model_static=model_static,
                optimizer=self.optimizer,
                optimizer_state=self.opt_state,
                batch=batch_sharded,
            )
            print(f"Step {i+1}/{self.trainer_config.num_steps}")
            print(f"Loss: {loss:.4f}")
            print(f"Accuracy: {accuracy:.4f}")
            print("-" * 40)
        pass


if __name__ == "__main__":
    trainer_config = pyrallis.parse(config_class=TrainerConfig)
    trainer = Trainer(trainer_config)
    trainer.train()
