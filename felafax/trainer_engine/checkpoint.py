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
    """A simple wrapper for orbax checkpointing."""

    def __init__(self, path: str):
        if not path:
            raise ValueError("Checkpoint path cannot be empty")
        self.path = path
        os.makedirs(self.path, exist_ok=True)
        self.checkpointer = ocp.StandardCheckpointer()

    def save_pytree(self, pytree, prefix=None):
        """Save pytree of JAX arrays."""
        path = os.path.join(self.path, prefix) if prefix else self.path
        self.checkpointer.save(path, pytree, force=True)
        # TODO: Add sentinel file saying save worked.

    @classmethod
    def restore_pytree(cls, path, item):
        return ocp.StandardCheckpointer().restore(
            path, args=ocp.args.StandardRestore(item)
        )

    def save_json(self, data, name):
        """Save dictionary as JSON."""
        if self.path == "":
            return
        path = os.path.join(self.path, name)
        os.makedirs(path, exist_ok=True)
        with open(path, "w") as f:
            f.write(json.dumps(data, indent=4))

    @classmethod
    def load_json(cls, path):
        with open(path, "r") as f:
            return json.loads(f.read())

    @classmethod
    def get_shape_dtype_struct(cls, tree):
        return jax.tree_util.tree_map(ocp.utils.to_shape_dtype_struct, tree)


def save_checkpoint(model: LlamaForCausalLM, path: str, step: int = None):
    """Save model checkpoint using Orbax."""
    # Create and prepare directory
    os.makedirs(os.path.dirname(path), exist_ok=True)
    path = ocp.test_utils.erase_and_create_empty(path)

    # Partition model into params and static
    model_params, model_static = eqx.partition(model, eqx.is_array)

    checkpointer = Checkpointer(path)
    prefix = f"step_{step}" if step is not None else None

    checkpointer.save_pytree(model_params, prefix=prefix)
    # TODO: Save llama config as JSON.


def load_checkpoint(
    model_name: str, path: Optional[str] = None, save_converted: bool = False
) -> LlamaForCausalLM:
    """Loads checkpoint, either from local storage using Orbax or downloads from HF.

    Args:
        model_name: Name of HF model (e.g. 'meta-llama/Llama-2-7b') or path to local checkpoint
        path: Optional path to save converted checkpoint. If None, uses a temporary directory
        save_converted: Whether to save HF checkpoint in Orbax format after conversion

    Returns:
        Loaded model
    """
    model = _load_from_hf(model_name)

    if save_converted and path is not None:
        save_checkpoint(model, path)
        print(f"Converted HF checkpoint and saved it in Orbax format at: {path}")

    # TODO: When loading from path, load llama config fro JSON file create class and then load params.
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


def _load_from_hf(path: str) -> LlamaForCausalLM:
    """Downloads and converts HuggingFace model to Equinox model."""
    # Load HF model
    hf_model = HFLlamaForCausalLM.from_pretrained(
        path, torch_dtype=torch.float32
    )

    # Create config and initialize Equinox model
    config = create_llama_config_from_hf_model(hf_model)
    eqx_model = LlamaForCausalLM(config)

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

    return eqx_model
