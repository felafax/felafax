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
from typing import Optional, Tuple
from jaxtyping import PyTree


class Checkpointer:
    def __init__(self, checkpoint_dir: str):
        if not checkpoint_dir:
            raise ValueError("Checkpoint directory cannot be empty")

        self.checkpoint_dir = checkpoint_dir
        self.options = ocp.CheckpointManagerOptions(
            max_to_keep=2,
            save_interval_steps=2,
            create=True,
            enable_async_checkpointing=False,
        )

        self.checkpoint_mgr = ocp.CheckpointManager(
            directory=self.checkpoint_dir,
            options=self.options,
            item_names=["model_pytree", "model_config"],
        )

    def save_checkpoint(
        self, model: eqx.Module, model_config: LlamaConfig, step: int = 0
    ):
        """Save model checkpoint using the provided Checkpointer."""
        model_pytree, _ = eqx.partition(model, eqx.is_array)
        self.checkpoint_mgr.save(
            step=step,
            args=ocp.args.Composite(
                model_pytree=ocp.args.StandardSave(model_pytree),
                model_config=ocp.args.JsonSave(model_config.to_dict()),
            ),
        )

    def restore_checkpoint(
        self, model_abstract_pytree: Optional[PyTree] = None
    ) -> Tuple[eqx.Module, LlamaConfig]:
        """Restore model checkpoint using the provided Checkpointer."""
        restored = self.checkpoint_mgr.restore(
            step=self.checkpoint_mgr.latest_step(),
            args=ocp.args.Composite(
                model_pytree=ocp.args.StandardRestore(model_abstract_pytree),
                model_config=ocp.args.JsonRestore(),
            ),
        )
        # Construct model using restored model config.
        model_config = LlamaConfig(**restored.model_config)
        model = LlamaForCausalLM(model_config)
        model_params, model_static = eqx.partition(model, eqx.is_array)
        model_params = jax.tree_util.tree_map(
            lambda x, y: y,
            model_params,
            restored.model_pytree,
        )

        # Combine restored model parameters with model static.
        model = eqx.combine(model_params, model_static)
        return model, model_config

    def wait_until_finished(self):
        """Wait for any async operations to complete."""
        self.checkpoint_mgr.wait_until_finished()

    @property
    def directory(self):
        return self.checkpoint_mgr.directory

    @classmethod
    def get_abstract_pytree(cls, tree):
        return jax.tree_util.tree_map(ocp.utils.to_shape_dtype_struct, tree)


def save_checkpoint(
    model: LlamaForCausalLM,
    model_config: LlamaConfig,
    checkpointer: Checkpointer,
    step: int,
):
    """Save model checkpoint using the provided Checkpointer.

    Args:
        model: The model to save
        model_config: The model configuration
        checkpointer: An instance of Checkpointer to manage saving
        step: Step number for the checkpoint
        metrics: Optional metrics dictionary for checkpointing
    """
    model_params, _ = eqx.partition(model, eqx.is_array)
    checkpointer.save_checkpoint(model, model_config, step)


def load_model(
    model_name: str,
) -> Tuple[LlamaForCausalLM, LlamaConfig]:
    """Loads checkpoint, either from local storage using Orbax or downloads from HF.

    Args:
        model_name: Name of HF model (e.g. 'meta-llama/Llama-2-7b') or path to local checkpoint

    Returns:
        tuple: (model, model_config)
    """
    model, model_config = _load_from_hf(model_name)
    return model, model_config


def load_model_or_checkpoint(
    model_name: str,
    checkpointer: Checkpointer,
) -> Tuple[LlamaForCausalLM, LlamaConfig]:
    """Loads checkpoint, either from local storage using Orbax or downloads from HF.

    Args:
        model_name: Name of HF model (e.g. 'meta-llama/Llama-2-7b') or path to local checkpoint
        checkpointer: An instance of Checkpointer to manage loading

    Returns:
        tuple: (model, model_config)
    """
    has_checkpoints = len(checkpointer.checkpoint_mgr.all_steps()) > 0
    if has_checkpoints:
        model, model_config = checkpointer.restore_checkpoint()
        return model, model_config

    model, model_config = load_model(model_name)
    return model, model_config


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

def _torch_to_jax(tensor):
    return jnp.array(tensor.detach().numpy())


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
        _torch_to_jax(hf_model.model.embed_tokens.weight),
    )
    eqx_model = eqx.tree_at(
        lambda t: t.model.norm.weight,
        eqx_model,
        _torch_to_jax(hf_model.model.norm.weight),
    )
    eqx_model = eqx.tree_at(
        lambda t: t.lm_head.weight,
        eqx_model,
        _torch_to_jax(hf_model.lm_head.weight),
    )

    # Copy layer weights
    for i, layer in enumerate(eqx_model.model.layers):
        hf_layer = hf_model.model.layers[i]
        eqx_model = eqx.tree_at(
            lambda t: t.model.layers[i].self_attn.q_proj.weight,
            eqx_model,
            _torch_to_jax(hf_layer.self_attn.q_proj.weight),
        )
        eqx_model = eqx.tree_at(
            lambda t: t.model.layers[i].self_attn.k_proj.weight,
            eqx_model,
            _torch_to_jax(hf_layer.self_attn.k_proj.weight),
        )
        eqx_model = eqx.tree_at(
            lambda t: t.model.layers[i].self_attn.v_proj.weight,
            eqx_model,
            _torch_to_jax(hf_layer.self_attn.v_proj.weight),
        )
        eqx_model = eqx.tree_at(
            lambda t: t.model.layers[i].self_attn.o_proj.weight,
            eqx_model,
            _torch_to_jax(hf_layer.self_attn.o_proj.weight),
        )
        eqx_model = eqx.tree_at(
            lambda t: t.model.layers[i].mlp.gate_proj.weight,
            eqx_model,
            _torch_to_jax(hf_layer.mlp.gate_proj.weight),
        )
        eqx_model = eqx.tree_at(
            lambda t: t.model.layers[i].mlp.up_proj.weight,
            eqx_model,
            _torch_to_jax(hf_layer.mlp.up_proj.weight),
        )
        eqx_model = eqx.tree_at(
            lambda t: t.model.layers[i].mlp.down_proj.weight,
            eqx_model,
            _torch_to_jax(hf_layer.mlp.down_proj.weight),
        )
        eqx_model = eqx.tree_at(
            lambda t: t.model.layers[i].input_layernorm.weight,
            eqx_model,
            _torch_to_jax(hf_layer.input_layernorm.weight),
        )
        eqx_model = eqx.tree_at(
            lambda t: t.model.layers[i].post_attention_layernorm.weight,
            eqx_model,
            _torch_to_jax(hf_layer.post_attention_layernorm.weight),
        )

    return eqx_model, model_config
