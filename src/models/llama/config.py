from dataclasses import dataclass
from typing import Optional, Dict, Any
from ...core.base_classes import ModelConfig


@dataclass
class LLaMAConfig(ModelConfig):
    """Configuration for LLaMA models."""

    vocab_size: int = 32000
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 32
    hidden_act: str = "silu"
    max_position_embeddings: int = 2048
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    tie_word_embeddings: bool = False
    rope_theta: float = 10000.0

    # LoRA-specific configuration
    use_lora: bool = False
    lora_rank: int = 8
    lora_alpha: float = 16
    lora_dropout: float = 0.1

    @classmethod
    def from_pretrained(cls, model_name: str) -> "LLaMAConfig":
        """Create config from predefined model sizes."""
        if "405b" in model_name.lower():
            return cls(
                vocab_size=128256,
                hidden_size=16384,
                intermediate_size=53248,
                num_hidden_layers=126,
                num_attention_heads=128,
                num_key_value_heads=8,
                max_position_embeddings=8192,
                rope_theta=5e5,
                rms_norm_eps=1e-5,
            )
        elif "70b" in model_name.lower():
            return cls(
                vocab_size=128256,
                hidden_size=8192,
                intermediate_size=28672,
                num_hidden_layers=80,
                num_attention_heads=64,
                num_key_value_heads=8,
                max_position_embeddings=8192,
                rope_theta=1e6,
            )
        elif "8b" in model_name.lower():
            return cls(
                vocab_size=128256,
                hidden_size=4096,
                intermediate_size=14336,
                num_hidden_layers=32,
                num_attention_heads=32,
                num_key_value_heads=8,
                max_position_embeddings=8192,
                rope_theta=1e6,
            )
        else:
            raise ValueError(f"Unknown model size in name: {model_name}")


class ModelRegistry:
    """Registry for LLaMA model configurations."""

    _MODELS = {
        "llama-3.1-8b": {
            "hf_name": "meta-llama/Llama-3.1-8b-hf",
            "config": LLaMAConfig.from_pretrained("8b"),
        },
        "llama-3.1-70b": {
            "hf_name": "meta-llama/Llama-3.1-70b-hf",
            "config": LLaMAConfig.from_pretrained("70b"),
        },
        "llama-3.1-405b": {
            "hf_name": "meta-llama/Llama-3.1-405b-hf",
            "config": LLaMAConfig.from_pretrained("405b"),
        },
    }

    @classmethod
    def get_config(cls, model_name: str) -> LLaMAConfig:
        """Get configuration for a specific model."""
        if model_name not in cls._MODELS:
            raise ValueError(f"Unknown model: {model_name}")
        return cls._MODELS[model_name]["config"]

    @classmethod
    def get_hf_name(cls, model_name: str) -> str:
        """Get HuggingFace model name."""
        if model_name not in cls._MODELS:
            raise ValueError(f"Unknown model: {model_name}")
        return cls._MODELS[model_name]["hf_name"]
