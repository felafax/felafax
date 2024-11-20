from src.felafax.trainer_engine.models.llama3.jax.model import (
    LlamaConfig,
    LlamaForCausalLM,
)
from src.felafax.trainer_engine.trainer import Trainer, TrainerConfig
import jax.numpy as jnp


def _get_dummy_data(trainer_config):
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


def dummy_data_loader(trainer_config):
    while True:
        input_ids, attention_mask, position_ids = _get_dummy_data(trainer_config)
        yield {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }


def test_llama_trainer():
    """Tests the end-to-end run of the trainer without checkpoing."""
    # Create a tiny model configuration using LlamaConfig
    model_config = LlamaConfig(
        model_name="tiny",
        vocab_size=100,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        attention_bias=False,
    )

    model = LlamaForCausalLM(model_config)

    trainer_config = TrainerConfig(
        seq_length=16,
        batch_size=8,
        num_steps=2,
        num_tpus=4,
    )
    # If we don't provide a checkpointer, the trainer should not save any checkpoints.
    assert trainer_config.checkpointer is None

    # Create dummy train and validation dataloaders
    train_dataloader = dummy_data_loader(trainer_config)
    val_dataloader = dummy_data_loader(trainer_config)

    # Initialize the trainer with the model and data loaders
    trainer = Trainer(
        trainer_config=trainer_config,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
    )
    trainer.model = model
    trainer.model_config = model_config

    # Run training
    trainer.train()
