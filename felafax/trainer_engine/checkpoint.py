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


def torch_to_jax(tensor):
    return jnp.array(tensor.detach().numpy())


class Checkpointer(object):
    """A simple wrapper for orbax checkpointing."""

    def __init__(self, path):
        self.path = path
        self.checkpointer = ocp.StandardCheckpointer()
        if self.path != "":
            os.makedirs(self.path, exist_ok=True)

    def save_pytree(self, pytree, prefix=None):
        """Save pytree of JAX arrays."""
        if self.path == "":
            return
        if prefix is None:
            path = self.path
        else:
            path = os.path.join(self.path, prefix)

        self.checkpointer.save(path, pytree, force=True)
        # Create a commit_success.txt file to indicate that the checkpoint is
        # saved successfully. This is a workaround for orbax so that locally
        # saved checkpoint can be restored when copied to Google cloud storage.
        with open(os.path.join(path, "commit_success.txt"), "w") as f:
            pass

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
    """Save model checkpoint using Orbax.

    Args:
        model: Model to save
        path: Directory to save checkpoint
        step: Optional step number to include in name
    """
    # Partition model into params and static
    model_params, model_static = eqx.partition(model, eqx.is_array)

    checkpointer = Checkpointer(path)
    prefix = f"step_{step}" if step is not None else None

    # Save params pytree
    checkpointer.save_pytree(model_params, prefix=prefix)

    # Save static config as JSON
    checkpointer.save_json(
        {
            "static": str(model_static)
        },  # Convert static part to string representation
        name=f"{prefix}/static.json" if prefix else "static.json",
    )


def load_checkpoint(
    model_name: str, path: str = None, save_converted: bool = False
) -> LlamaForCausalLM:
    """Loads checkpoint, either from local storage using Orbax or downloads from HF.

    Args:
        model_name: Name of HF model (e.g. 'meta-llama/Llama-2-7b') or path to local checkpoint
        path: Optional path to save/load local checkpoint. If None, uses model_name as path
        save_converted: Whether to save HF checkpoint in Orbax format after conversion

    Returns:
        Loaded model
    """
    # Use model_name as path if no path specified
    checkpoint_path = path

    # First try loading Orbax checkpoint from local storage
    try:
        # Check if checkpoint exists by looking for success sentinel
        if checkpoint_path and os.path.exists(
            f"{checkpoint_path}/commit_success.txt"
        ):
            # Create empty model for structure
            config = create_llama_config_from_hf_model(
                HFLlamaForCausalLM.from_pretrained(model_name)
            )
            empty_model = LlamaForCausalLM(config)

            # Partition empty model to get structure
            empty_params, model_static = eqx.partition(empty_model, eqx.is_array)

            # Restore params
            checkpointer = Checkpointer(checkpoint_path)
            restored_params = checkpointer.restore_pytree(
                checkpoint_path, empty_params
            )

            # Combine restored params with static
            return eqx.combine(restored_params, model_static)
        else:
            raise FileNotFoundError(
                "No valid checkpoint found at specified path"
            )

    except Exception as e:
        print(f"Loading from local checkpoint failed: {str(e)}")
        print("Falling back to loading from HF checkpoint...")

        # Fall back to loading from HF checkpoint
        model = _load_from_hf(model_name)

        # Optionally save the converted model in Orbax format
        if save_converted:
            save_checkpoint(model, checkpoint_path)
            print(
                f"Converted HF checkpoint saved in Orbax format at: {checkpoint_path}"
            )

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
