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


def torch_to_jax(tensor):
    return jnp.array(tensor.detach().numpy())


class Checkpointer(object):
    """Wraps Orbax checkpointing using CheckpointManager."""

    def __init__(self, path: str):
        if not path:
            raise ValueError("Checkpoint path cannot be empty")
        self.path = path
        os.makedirs(self.path, exist_ok=True)

        # Create CheckpointManagerOptions with the desired options
        self.options = ocp.CheckpointManagerOptions(
            max_to_keep=3,  # Keep only latest 3 checkpoints
            save_interval_steps=2,  # Save every 2 steps
            keep_period=10,  # Additionally keep checkpoints every 10 steps
            create=True,
            enable_async_checkpointing=False,
        )

        # Define a checkpointer for PyTree objects
        checkpointers = {"model": ocp.PyTreeCheckpointer()}

        # Create the CheckpointManager
        self.checkpoint_manager = ocp.CheckpointManager(
            directory=self.path,
            checkpointers=checkpointers,
            options=self.options,
        )

    def save_pytree(self, pytree, step: int, metrics: dict = None):
        """Save pytree of JAX arrays at a given step."""
        items = {"model": pytree}
        self.checkpoint_manager.save(step=step, items=items, metrics=metrics)
        sentinel_path = os.path.join(self.path, "checkpoint_success.txt")
        with open(sentinel_path, "w") as f:
            f.write(f"Checkpoint successfully saved at: {self.path}\n")

    def restore_pytree(self, step: Optional[int] = None):
        if step is None:
            # Restore latest checkpoint
            step = self.checkpoint_manager.latest_step()
        restored = self.checkpoint_manager.restore(step, items={"model": None})
        return restored["model"]

    def wait_until_finished(self):
        self.checkpoint_manager.wait_until_finished()

    def save_json(self, data: dict, name: str) -> None:
        """Save dictionary as JSON.

        Args:
            data: Dictionary containing JSON-serializable data
            name: Name of the file to save the JSON data
        """
        path = os.path.join(self.path, name)
        with open(path, "w") as f:
            json.dump(data, f, indent=4)

    def load_json(self, name: str):
        path = os.path.join(self.path, name)
        with open(path, "r") as f:
            return json.load(f)

    @classmethod
    def get_abstract_pytree(cls, tree):
        return jax.tree_util.tree_map(ocp.utils.to_shape_dtype_struct, tree)


def save_checkpoint(
    model: LlamaForCausalLM,
    model_config: LlamaConfig,
    checkpointer: Checkpointer,
    step: int,
    metrics: dict = None,
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
    checkpointer.save_pytree(model_params, step=step, metrics=metrics)
    checkpointer.save_json(model_config.to_dict(), name="model_config.json")


def load_checkpoint(
    model_name: str,
    checkpointer: Optional[Checkpointer] = None,
    save_converted: bool = False,
) -> Tuple[LlamaForCausalLM, LlamaConfig]:
    """Loads checkpoint, either from local storage using Orbax or downloads from HF.

    Args:
        model_name: Name of HF model (e.g. 'meta-llama/Llama-2-7b') or path to local checkpoint
        checkpointer: An instance of Checkpointer to manage loading
        save_converted: Whether to save HF checkpoint in Orbax format after conversion

    Returns:
        tuple: (model, model_config)
    """
    if checkpointer:
        config_exists = os.path.exists(
            os.path.join(checkpointer.path, "model_config.json")
        )
        has_checkpoints = checkpointer.checkpoint_manager.all_steps()
        if config_exists and has_checkpoints:
            # Load config from JSON
            config_data = checkpointer.load_json("model_config.json")
            model_config = LlamaConfig(**config_data)
            model = LlamaForCausalLM(model_config)

            # Restore model parameters
            model_params = checkpointer.restore_pytree()
            model = eqx.combine(model_params, model)
            print(f"Loaded model from checkpoint at: {checkpointer.path}")
            return model, model_config

    # If checkpoint not found or checkpointer is None, load from HF
    model, model_config = _load_from_hf(model_name)

    if save_converted and checkpointer:
        # Save the downloaded model using the checkpointer
        save_checkpoint(model, model_config, checkpointer, step=0)
        print(
            f"Converted HF checkpoint and saved it in Orbax format at: {checkpointer.path}"
        )

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
