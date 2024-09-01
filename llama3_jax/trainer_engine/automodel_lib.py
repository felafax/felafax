import os
from typing import Tuple, Union, Dict

from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoTokenizer
from felafax.llama_config import create_llama_model, Llama3_1_8B_Configurator, Llama3_1_70B_Configurator, LlamaTestConfigurator
from felafax import llama_model
import jax.numpy as jnp

LlamaConfigType = Union[Dict, Llama3_1_8B_Configurator,
                        Llama3_1_70B_Configurator, LlamaTestConfigurator]

MODEL_NAME_TO_DOWNLOAD_CONFIG = {
    "llama-3.1-8B-JAX": {
        "hf_model_name": "meta-llama/Meta-Llama-3.1-8B",
        "felafax_model_name": "felafax/llama-3.1-8B-JAX",
        "chkpt_filename": "llama3.1_8b_serialized.flax",
    },
}


class AutoJAXModelForCausalLM:

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        huggingface_token: str,
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

        hf_config = AutoConfig.from_pretrained(
            download_config["hf_model_name"], token=huggingface_token)

        tokenizer = AutoTokenizer.from_pretrained(
            download_config["hf_model_name"],
            token=huggingface_token,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model_path = snapshot_download(
            repo_id=download_config["felafax_model_name"],
            token=huggingface_token,
        )
        model_path = os.path.join(model_path,
                                  download_config["chkpt_filename"])

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
