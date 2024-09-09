import os
from typing import Dict, Tuple, Union, Optional

import jax.numpy as jnp
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoTokenizer

from . import llama_model
from .llama_config import (Llama3_1_8B_Configurator,
                            Llama3_1_70B_Configurator, LlamaTestConfigurator,
                            create_llama_model)

LlamaConfigType = Union[Dict, Llama3_1_8B_Configurator,
                        Llama3_1_70B_Configurator, LlamaTestConfigurator]

MODEL_NAME_TO_DOWNLOAD_CONFIG = {
    "llama-3.1-8B-JAX": {
        "hf_model_name": "meta-llama/Meta-Llama-3.1-8B",
        "felafax_model_name": "felafax/llama-3.1-8B-JAX",
        "chkpt_filename": "llama3.1_8b_serialized.flax",
    },
    "llama-3.1-8B-Instruct-JAX": {
        "hf_model_name": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "felafax_model_name": "felafax/llama-3.1-8B-Instruct-JAX",
        "chkpt_filename": "llama-3.1-8B-Instruct-JAX.flax",
    },
    "colab-llama-3.1-8B-Instruct-JAX": {
        "hf_model_name": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "felafax_model_name": "felafax/colab-llama-3.1-8B-Instruct-JAX",
        "chkpt_filename": "llama-3.1-8B-Instruct-JAX.flax",
    },
}


class AutoJAXModelForCausalLM:

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        huggingface_token: Optional[str] = None,
        **kwargs,
    ) -> Tuple[str, llama_model.CausalLlamaModule, LlamaConfigType,
               AutoTokenizer]:
        """Downloads the model from HF and returns the downloaded model path, model, llama config, and tokenizer."""

        print(f"Downloading model {model_name}...")
        try:
            download_config = MODEL_NAME_TO_DOWNLOAD_CONFIG[model_name]
        except KeyError:
            raise ValueError(
                f"Invalid model name: {model_name}. "
                f"Available models are: {', '.join(MODEL_NAME_TO_DOWNLOAD_CONFIG.keys())}"
            )

        model_dir = snapshot_download(
            repo_id=download_config["felafax_model_name"],
            token=huggingface_token if huggingface_token else None,
        )
        model_path = os.path.join(model_dir, download_config["chkpt_filename"])

        if huggingface_token:
            tokenizer = AutoTokenizer.from_pretrained(
                download_config["hf_model_name"], token=huggingface_token)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_dir)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print(f"{model_name} was downloaded to {model_path}.")

        # Create LlamaFactory and model
        llama_model_configurator = create_llama_model(model_name)
        llama_model_config = llama_model_configurator.get_model_config()
        llama_model_hf_config = llama_model_configurator.get_hf_pretrained_config(
            llama_model_config)

        model = llama_model.CausalLlamaModule(
            llama_model_hf_config,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
        )

        return model_path, model, llama_model_configurator, tokenizer
