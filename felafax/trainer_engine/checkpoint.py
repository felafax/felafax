"""Loads and saves checkpoint."""

import os
import json
import jax
import jax.numpy as jnp
import equinox as eqx
import torch
import orbax.checkpoint as ocp
from transformers import LlamaForCausalLM as HFLlamaForCausalLM
from felafax.trainer_engine.models.llama3.jax.model import (
    LlamaConfig,
    LlamaForCausalLM,
)
from typing import Optional


def torch_to_jax(tensor):
    return jnp.array(tensor.detach().numpy())


class Checkpointer(object):
    """Wraps Orbax checkpointing."""

    def __init__(self, path: str):
        if not path:
            raise ValueError("Checkpoint path cannot be empty")
        self.path = path
        os.makedirs(self.path, exist_ok=True)

        handler = ocp.StandardCheckpointHandler()
        options = ocp.CheckpointerOptions(enable_async=False)
        self.checkpointer = ocp.Checkpointer(handler, options=options)

    def save_pytree(self, pytree, prefix=None):
        """Save pytree of JAX arrays."""
        path = os.path.join(self.path, prefix) if prefix else self.path
        self.checkpointer.save(path, pytree, force=True)

        # Add sentinel file indicating successful save
        sentinel_path = os.path.join(
            os.path.dirname(path), "checkpoint_success.txt"
        )
        with open(sentinel_path, "w") as f:
            f.write(f"Checkpoint successfully saved at: {path}\n")

    @classmethod
    def restore_pytree(cls, path, item):
        return ocp.StandardCheckpointer().restore(
            path, args=ocp.args.StandardRestore(item)
        )

    def save_json(self, data: dict, name: str) -> None:
        """Save dictionary as JSON.

        Args:
            data: Dictionary containing JSON-serializable data
            name: Name of the file/directory to save the JSON data
        """
        path = os.path.join(self.path, name)
        os.makedirs(path, exist_ok=True)
        with open(path, "w") as f:
            f.write(json.dumps(data, indent=4))

    @classmethod
    def load_json(cls, path):
        with open(path, "r") as f:
            return json.loads(f.read())

    @classmethod
    def get_abstract_pytree(cls, tree):
        return jax.tree_util.tree_map(ocp.utils.to_shape_dtype_struct, tree)


def save_checkpoint(
    model: LlamaForCausalLM,
    model_config: LlamaConfig,
    checkpoint_dir: str,
    step: int = None,
):
    """Save model checkpoint using Orbax.

    Args:
        model: The model to save
        path: Path to save the checkpoint
        model_config: The model configuration
        step: Optional step number to include in checkpoint name
    """
    model_params, model_static = eqx.partition(model, eqx.is_array)

    os.makedirs(os.path.dirname(checkpoint_dir), exist_ok=True)
    checkpoint_dir = ocp.test_utils.erase_and_create_empty(checkpoint_dir)

    checkpointer = Checkpointer(checkpoint_dir)
    checkpointer.save_pytree(
        model_params, prefix=(f"step_{step}" if step else None)
    )

    config_dict = model_config.to_dict()
    checkpointer.save_json(config_dict, name="model_config.json")


def load_checkpoint(
    model_name: str,
    checkpoint_dir: Optional[str] = None,
    save_converted: bool = False,
) -> tuple[LlamaForCausalLM, LlamaConfig]:
    """Loads checkpoint, either from local storage using Orbax or downloads from HF.

    Args:
        model_name: Name of HF model (e.g. 'meta-llama/Llama-2-7b') or path to local checkpoint
        path: Optional path to save converted checkpoint. If None, uses a temporary directory
        save_converted: Whether to save HF checkpoint in Orbax format after conversion

    Returns:
        tuple: (model, model_config)
    """
    sentinel_path = os.path.join(checkpoint_dir, "checkpoint_success.txt")
    config_path = os.path.join(checkpoint_dir, "model_config.json")

    if (
        checkpoint_dir
        and os.path.exists(config_path)
        and os.path.exists(sentinel_path)
    ):
        # Load config from JSON
        config_data = Checkpointer.load_json(config_path)
        model_config = LlamaConfig(**config_data)
        model = LlamaForCausalLM(model_config)

        params_structure = Checkpointer.get_abstract_pytree(
            eqx.filter(model, eqx.is_array)
        )
        model_params = Checkpointer.restore_pytree(
            checkpoint_dir, params_structure
        )

        model = eqx.combine(model_params, model)
        print(f"Loaded model from checkpoint at: {checkpoint_dir}")
        return model, model_config
    else:
        model, model_config = _load_from_hf(model_name)

        if save_converted and checkpoint_dir is not None:
            save_checkpoint(model, model_config, checkpoint_dir)
            print(
                f"Converted HF checkpoint and saved it in Orbax format at: {checkpoint_dir}"
            )

        return model, model_config
    return model


def create_llama_config_from_hf_model(hf_model) -> LlamaConfig:
    """Creates Equinox config from Hugging Face model config."""
    return LlamaConfig(
        vocab_size=hf_model.config.vocab_size,
        hidden_size=hf_model.config.hidden_size,
        intermediate_size=hf_model.config.intermediate_size,
        num_hidden_layers=hf_model.config.num_hidden_layers,
        num_attention_heads=hf_model.config.num_attention_heads,
        num_key_value_heads=hf_model.config.num_key_value_heads,
        max_position_embeddings=hf_model.config.max_position_embeddings,
        rms_norm_eps=hf_model.config.rms_norm_eps,
        rope_theta=hf_model.config.rope_theta,
        attention_bias=hf_model.config.attention_bias,
    )


def _load_from_hf(model_name: str) -> tuple[LlamaForCausalLM, LlamaConfig]:
    """Downloads and converts HuggingFace model to Equinox model.

    Args:
        model_name: Name of the HuggingFace model to load

    Returns:
        tuple: (eqx_model, model_config)
    """
    # Load HF model
    hf_model = HFLlamaForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32
    )

    # Create config and initialize Equinox model
    model_config = create_llama_config_from_hf_model(hf_model)
    eqx_model = LlamaForCausalLM(model_config)

    # Copy weights
    eqx_model = eqx.tree_at(
        lambda t: t.model.embed_tokens.weight,
        eqx_model,
        torch_to_jax(hf_model.model.embed_tokens.weight),
    )
    eqx_model = eqx.tree_at(
        lambda t: t.model.norm.weight,
        eqx_model,
        torch_to_jax(hf_model.model.norm.weight),
    )
    eqx_model = eqx.tree_at(
        lambda t: t.lm_head.weight,
        eqx_model,
        torch_to_jax(hf_model.lm_head.weight),
    )

    # Copy layer weights
    for i, layer in enumerate(eqx_model.model.layers):
        hf_layer = hf_model.model.layers[i]
        eqx_model = eqx.tree_at(
            lambda t: t.model.layers[i].self_attn.q_proj.weight,
            eqx_model,
            torch_to_jax(hf_layer.self_attn.q_proj.weight),
        )
        eqx_model = eqx.tree_at(
            lambda t: t.model.layers[i].self_attn.k_proj.weight,
            eqx_model,
            torch_to_jax(hf_layer.self_attn.k_proj.weight),
        )
        eqx_model = eqx.tree_at(
            lambda t: t.model.layers[i].self_attn.v_proj.weight,
            eqx_model,
            torch_to_jax(hf_layer.self_attn.v_proj.weight),
        )
        eqx_model = eqx.tree_at(
            lambda t: t.model.layers[i].self_attn.o_proj.weight,
            eqx_model,
            torch_to_jax(hf_layer.self_attn.o_proj.weight),
        )
        eqx_model = eqx.tree_at(
            lambda t: t.model.layers[i].mlp.gate_proj.weight,
            eqx_model,
            torch_to_jax(hf_layer.mlp.gate_proj.weight),
        )
        eqx_model = eqx.tree_at(
            lambda t: t.model.layers[i].mlp.up_proj.weight,
            eqx_model,
            torch_to_jax(hf_layer.mlp.up_proj.weight),
        )
        eqx_model = eqx.tree_at(
            lambda t: t.model.layers[i].mlp.down_proj.weight,
            eqx_model,
            torch_to_jax(hf_layer.mlp.down_proj.weight),
        )
        eqx_model = eqx.tree_at(
            lambda t: t.model.layers[i].input_layernorm.weight,
            eqx_model,
            torch_to_jax(hf_layer.input_layernorm.weight),
        )
        eqx_model = eqx.tree_at(
            lambda t: t.model.layers[i].post_attention_layernorm.weight,
            eqx_model,
            torch_to_jax(hf_layer.post_attention_layernorm.weight),
        )

    return eqx_model, model_config
