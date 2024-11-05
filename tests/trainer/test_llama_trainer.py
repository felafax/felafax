import pytest
import jax.numpy as jnp
import optax
from transformers.configuration_utils import PretrainedConfig
from dataclasses import dataclass
from ml_collections import ConfigDict
from copy import deepcopy
from felafax.trainer_engine.models.llama3.jax.model import LlamaModel

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
    llama_config = TestLlamaConfig()
    llama_pretrained_hf_config = llama_config.get_pretrained_hf_config
    llama_model = LlamaModel(llama_pretrained_hf_config)
    
    
    pass
