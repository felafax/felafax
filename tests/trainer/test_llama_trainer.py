from felafax.trainer_engine.models.llama3.jax.model import (
    LlamaConfig,
    LlamaForCausalLM,
)
from felafax.trainer_engine.trainer import Trainer, TrainerConfig


def test_llama_trainer():
    """Tests the e2e run of the trainer with a tiny model. REQUIRES 8 TPUs."""
    # Create a tiny config using our own LlamaConfig class
    config = LlamaConfig(
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

    # Initialize model with our config
    model = LlamaForCausalLM(config)

    # Create trainer config (no model_path needed)
    trainer_config = TrainerConfig(
        seq_length=16, batch_size=2, num_steps=2, num_tpus=8
    )

    # Initialize trainer with our model
    trainer = Trainer(trainer_config, model=model)

    # Run training
    trainer.train()
