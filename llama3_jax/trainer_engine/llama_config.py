from copy import copy, deepcopy
from typing import Union

from jax.sharding import PartitionSpec as PS
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_flax_outputs import (FlaxBaseModelOutput,
                                                FlaxCausalLMOutput)

from . import utils


class LlamaFactory:
    """Creates and configures LLaMA model for various sizes."""

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
        self.model_config = utils.create_config_dict(self.base_config)

    def get_model_config(self):
        return self.model_config

    def get_hf_pretrained_config(self, config):
        """Apply updates on top of standard base model config."""
        updated_config = deepcopy(config)
        return PretrainedConfig.from_dict(updated_config)

    def get_partition_rules(self):
        """Rules for partitioning llama model across TPU cores."""
        return (
            # embeddings
            ("transformer/wte/embedding", PS("mp", "fsdp")),
            # atention
            ("attention/(wq|wk|wv)/kernel", PS("fsdp", "mp")),
            ("attention/wo/kernel", PS("mp", "fsdp")),
            # mlp
            ("feed_forward/w1/kernel", PS("fsdp", "mp")),
            ("feed_forward/w2/kernel", PS("mp", "fsdp")),
            ("feed_forward/w3/kernel", PS("fsdp", "mp")),
            # layer norms
            ("attention_norm/kernel", PS(None)),
            ("ffn_norm/kernel", PS(None)),
            # output head
            ("transformer/ln_f/kernel", PS(None)),
            ("lm_head/kernel", PS("fsdp", "mp")),
            # Add rules for batch-specific parameters (if any)
            ("batch_stats/.*", PS("dp")),
            # Keep the catch-all rule at the end
            (".*", PS(None)),
        )

    def get_rng_keys(self):
        return ("params", "dropout", "fcm")


class LlamaTestConfigurator(LlamaFactory):

    def __init__(self):
        super().__init__()
        # LlamaTest uses the base configuration without any modifications
        # All settings are inherited from LlamaFactory


class Llama3_1_8B_Configurator(LlamaFactory):

    def __init__(self):
        super().__init__()
        self.model_config.update({
            "base_model": "llama3.1_8b",
            "vocab_size": 128256,
            "hidden_size": 4096,
            "intermediate_size": 14336,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "max_position_embeddings": 8192,
            "rms_norm_eps": 1e-5,
            "rope_theta": 5e5,
        })


class Llama3_1_70B_Configurator(LlamaFactory):

    def __init__(self):
        super().__init__()
        self.model_config.update({
            "base_model": "llama3.1_70b",
            "vocab_size": 128256,
            "hidden_size": 8192,
            "intermediate_size": 28672,
            "num_hidden_layers": 80,
            "num_attention_heads": 64,
            "num_key_value_heads": 8,
            "max_position_embeddings": 8192,
            "rms_norm_eps": 1e-5,
            "rope_theta": 5e5,
        })


def create_llama_model(
    model_name: str
) -> Union[LlamaTestConfigurator, Llama3_1_8B_Configurator,
           Llama3_1_70B_Configurator]:
    """Creates and returns the appropriate llama model."""
    if model_name in ("llama-3.1-8B-JAX", "llama-3.1-8B-Instruct-JAX", "colab-llama-3.1-8B-Instruct-JAX"):
        return Llama3_1_8B_Configurator()
    elif model_name in ("llama-3.1-70B-JAX", "llama-3.1-70B-Instruct-JAX"):
        return Llama3_1_70B_Configurator()
    elif model_name == "llama_test":
        return LlamaFactory()
    else:
        raise ValueError(f"[auto_lib] Invalid model name: {model_name}")
