from src.felafax.trainer_engine.models.llama3.jax.model import (
    LlamaConfig,
    LlamaForCausalLM,
)
from src.felafax.trainer_engine.trainer import Trainer, TrainerConfig
import jax.numpy as jnp
import jax


def _get_dummy_data(batch_size, seq_length):
    input_ids = jnp.zeros((batch_size, seq_length), dtype=jnp.int32)
    attention_mask = jnp.ones((batch_size, seq_length), dtype=jnp.int32)
    position_ids = jnp.repeat(
        jnp.arange(0, seq_length)[None, :],
        batch_size,
        axis=0,
    )
    return input_ids, attention_mask, position_ids


def dummy_data_loader(batch_size, seq_length):
    while True:
        input_ids, attention_mask, position_ids = _get_dummy_data(batch_size, seq_length)
        yield {
            "input_ids": input_ids,
            "labels": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }


def test_llama_trainer():
    """Tests the end-to-end run of the trainer without checkpointing."""
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

    # Initialize model directly instead of loading from HF
    key = jax.random.PRNGKey(0)
    model = LlamaForCausalLM(
        model_config,
        param_dtype=jnp.float32,
        compute_dtype=jnp.float32,
        key=key
    )

    trainer_config = TrainerConfig(
        model_name="",  # Empty because we're providing the model directly
        num_steps=2,
        num_tpus=jax.device_count(),
        use_lora=True,
        lora_rank=8,
        learning_rate=1e-3,
        base_dir="/tmp/test_trainer/",
        param_dtype="float32",
        compute_dtype="float32",
    )

    batch_size = 8
    seq_length = 16

    # Create dummy train and validation dataloaders
    train_dataloader = dummy_data_loader(batch_size, seq_length)
    val_dataloader = dummy_data_loader(batch_size, seq_length)

    # Initialize the trainer, passing the model and model_config
    trainer = Trainer(
        trainer_config=trainer_config,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        model=model,
        model_config=model_config
    )

    # If we don't provide a checkpointer, the trainer should not save any checkpoints.
    assert trainer.checkpointer is None

    # Run training
    trainer.train()
