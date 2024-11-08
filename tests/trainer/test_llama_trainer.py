import pytest
import jax.numpy as jnp
import optax
from transformers.configuration_utils import PretrainedConfig
from dataclasses import dataclass
from ml_collections import ConfigDict
from copy import deepcopy
from felafax.trainer_engine.models.llama3.jax.model import LlamaConfig, LlamaForCausalLM
from felafax.trainer_engine.trainer import Trainer, TrainerConfig

class TestLlamaConfig:

    def __init__(self):
        self.base_config = {
            "base_model": "llama_test",
            "vocab_size": 32000,
            "hidden_size": 3200,
            "intermediate_size": 8640,
            "num_hidden_layers": 26,
            "num_attention_heads": 32,
            "num_key_value_heads": 32,
            "initializer_range": 0.02,
            "rms_norm_eps": 1e-6,
            "max_position_embeddings": 2048,
            "rope_theta": 1e4,
            "embedding_dropout": 0.0,
            "attention_dropout": 0.0,
            "residue_dropout": 0.0,
        }
        self.model_config = ConfigDict(self.base_config)

    @property
    def get_model_config(self):
        return self.model_config

    @property
    def get_pretrained_hf_config(self):
        """Apply updates on top of standard base model config."""
        updated_config = deepcopy(self.get_model_config)
        return PretrainedConfig.from_dict(updated_config)

def test_llama_trainer():
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
        attention_bias=False
    )
    
    # Initialize model with our config
    model = LlamaForCausalLM(config)
    
    # Create trainer config (no model_path needed)
    trainer_config = TrainerConfig(
        seq_length=16,
        batch_size=2,
        num_steps=2,
        num_tpus=4
    )
    
    # Initialize trainer with our model
    trainer = Trainer(trainer_config, model=model)
    
    # Run training
    trainer.train()
