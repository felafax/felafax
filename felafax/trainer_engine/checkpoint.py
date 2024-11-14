"""Loads and saves checkpoint."""

import os
import json
import jax
import jax.numpy as jnp
import numpy as np

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
from jax.sharding import NamedSharding, PartitionSpec as PS
from jax.experimental import mesh_utils

from dataclasses import dataclass


@dataclass
class CheckpointerConfig:
    """Configuration for checkpointing"""

    checkpoint_dir: str
    max_to_keep: int = 2
    save_interval_steps: int = 10
    create: bool = True  # Create the checkpoint directory if it doesn't exist
    enable_async_checkpointing: bool = True


class Checkpointer:
    def __init__(self, config: CheckpointerConfig):
        if not config.checkpoint_dir:
            raise ValueError("Checkpoint directory cannot be empty")

        self.checkpoint_dir = config.checkpoint_dir
        self.options = ocp.CheckpointManagerOptions(
            max_to_keep=config.max_to_keep,
            save_interval_steps=config.save_interval_steps,
            create=config.create,
            enable_async_checkpointing=config.enable_async_checkpointing,
        )

        self.checkpoint_mgr = ocp.CheckpointManager(
            directory=self.checkpoint_dir,
            options=self.options,
            item_names=["model_pytree", "model_config"],
        )

    @classmethod
    def get_abstract_pytree(cls, tree):
        return jax.tree_util.tree_map(ocp.utils.to_shape_dtype_struct, tree)

    def save_checkpoint(
        self, model: eqx.Module, model_config: LlamaConfig, step: int = 0
    ):
        """Saves model checkpoint."""
        model_pytree, _ = eqx.partition(model, eqx.is_inexact_array)
        self.checkpoint_mgr.save(
            step=step,
            args=ocp.args.Composite(
                model_pytree=ocp.args.StandardSave(model_pytree),
                model_config=ocp.args.JsonSave(model_config.to_dict()),
            ),
        )

    def restore_checkpoint(self) -> Tuple[eqx.Module, LlamaConfig]:
        """Restores model checkpoint."""
        # Step 1: Restore the model_config first
        restored_config = self.checkpoint_mgr.restore(
            step=self.checkpoint_mgr.latest_step(),
            items=["model_config"],
            args=ocp.args.Composite(
                model_config=ocp.args.JsonRestore(),
            ),
        )
        model_config = LlamaConfig(**restored_config["model_config"])

        # Step 2: Construct the model and create the abstract pytree
        model = LlamaForCausalLM(model_config)
        model_params, model_static = eqx.partition(model, eqx.is_inexact_array)
        model_abstract_pytree = self.get_abstract_pytree(model_params)

        # Step 3: Restore the model parameters using the abstract pytree
        restored_params = self.checkpoint_mgr.restore(
            step=self.checkpoint_mgr.latest_step(),
            items=["model_pytree"],
            args=ocp.args.Composite(
                model_pytree=ocp.args.StandardRestore(model_abstract_pytree),
            ),
        )

        # Combine restored model parameters with model static
        model_params = restored_params["model_pytree"]
        model = eqx.combine(model_params, model_static)
        return model, model_config

    def wait_until_finished(self):
        """Wait for any async operations to complete."""
        self.checkpoint_mgr.wait_until_finished()

    @property
    def directory(self):
        return self.checkpoint_mgr.directory


def load_model(
    model_name: str,
) -> Tuple[LlamaForCausalLM, LlamaConfig]:
    """Loads model from Hugging Face.

    Args:
        model_name: Name of HF model (e.g. 'meta-llama/Llama-2-7b').

    Returns:
        tuple: (model, model_config)
    """
    model, model_config = load_llama_from_hf(model_name)
    return model, model_config


def load_checkpoint_or_model(
    model_name: str,
    checkpointer: Checkpointer,
) -> Tuple[LlamaForCausalLM, LlamaConfig]:
    """Loads checkpoint from local storage using Orbax or downloads from HF.

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


def _make_torch_to_jax():
    """Creates a closure that converts PyTorch tensors to JAX arrays with sharding annotations."""
    # Import here to avoid circular dependency
    from felafax.trainer_engine.trainer import get_mesh

    mesh = get_mesh(jax.device_count())

    def _torch_to_jax(tensor, sharding_spec):
        jax_array = jnp.array(tensor.detach().numpy())
        sharding = NamedSharding(mesh, sharding_spec)
        return jax.device_put(jax_array, sharding)

    return _torch_to_jax

# TODO(refactor): Move load model into models/llama.
def load_llama_from_hf(model_name: str) -> Tuple[LlamaForCausalLM, LlamaConfig]:
    """Downloads and converts Hugging Face model to Equinox model.

    Args:
        model_name: Name of the Hugging Face model to load

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

    torch_to_jax = _make_torch_to_jax()

    # Copy weights from HF model to Equinox model
    eqx_model = eqx.tree_at(
        lambda t: t.model.embed_tokens.weight,
        eqx_model,
        torch_to_jax(hf_model.model.embed_tokens.weight, PS(('mp', 'fsdp'))),
    )
    eqx_model = eqx.tree_at(
        lambda t: t.model.norm.weight,
        eqx_model,
        torch_to_jax(hf_model.model.norm.weight, PS()),
    )
    eqx_model = eqx.tree_at(
        lambda t: t.lm_head.weight,
        eqx_model,
        torch_to_jax(hf_model.lm_head.weight, PS(('fsdp', 'mp'))),
    )

    # Copy layer weights with appropriate sharding
    for i in range(len(eqx_model.model.layers)):
        hf_layer = hf_model.model.layers[i]

        # Self-attention weights
        eqx_model = eqx.tree_at(
            lambda t: t.model.layers[i].self_attn.q_proj.weight,
            eqx_model,
            torch_to_jax(hf_layer.self_attn.q_proj.weight, PS(('fsdp', 'mp'))),
        )
        eqx_model = eqx.tree_at(
            lambda t: t.model.layers[i].self_attn.k_proj.weight,
            eqx_model,
            torch_to_jax(hf_layer.self_attn.k_proj.weight, PS(('fsdp', 'mp'))),
        )
        eqx_model = eqx.tree_at(
            lambda t: t.model.layers[i].self_attn.v_proj.weight,
            eqx_model,
            torch_to_jax(hf_layer.self_attn.v_proj.weight, PS(('fsdp', 'mp'))),
        )
        eqx_model = eqx.tree_at(
            lambda t: t.model.layers[i].self_attn.o_proj.weight,
            eqx_model,
            torch_to_jax(hf_layer.self_attn.o_proj.weight, PS(('mp', 'fsdp'))),
        )

        # MLP weights
        eqx_model = eqx.tree_at(
            lambda t: t.model.layers[i].mlp.gate_proj.weight,
            eqx_model,
            torch_to_jax(hf_layer.mlp.gate_proj.weight, PS(('fsdp', 'mp'))),
        )
        eqx_model = eqx.tree_at(
            lambda t: t.model.layers[i].mlp.up_proj.weight,
            eqx_model,
            torch_to_jax(hf_layer.mlp.up_proj.weight, PS(('fsdp', 'mp'))),
        )
        eqx_model = eqx.tree_at(
            lambda t: t.model.layers[i].mlp.down_proj.weight,
            eqx_model,
            torch_to_jax(hf_layer.mlp.down_proj.weight, PS(('mp', 'fsdp'))),
        )

        # Layer norms
        eqx_model = eqx.tree_at(
            lambda t: t.model.layers[i].input_layernorm.weight,
            eqx_model,
            torch_to_jax(hf_layer.input_layernorm.weight, PS()),
        )
        eqx_model = eqx.tree_at(
            lambda t: t.model.layers[i].post_attention_layernorm.weight,
            eqx_model,
            torch_to_jax(hf_layer.post_attention_layernorm.weight, PS()),
        )

    return eqx_model, model_config

def save_model_to_hf(
    model: eqx.Module,
    model_config: LlamaConfig,
    output_dir: str,
    tokenizer_name: str = "meta-llama/Llama-2-7b-hf",
):
    """Converts Equinox model back to Hugging Face format and saves it.

    Args:
        model: Equinox LlamaForCausalLM model
        model_config: LlamaConfig used for the model
        output_dir: Directory where to save the Hugging Face model
        tokenizer_name: Name of the tokenizer to download and save
    """
    import torch
    import numpy as np
    from transformers import LlamaForCausalLM as HFLlamaForCausalLM
    from transformers import LlamaConfig as HFLlamaConfig
    from transformers import AutoTokenizer

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Create a Hugging Face config from Equinox config
    hf_config = HFLlamaConfig(
        vocab_size=model_config.vocab_size,
        hidden_size=model_config.hidden_size,
        intermediate_size=model_config.intermediate_size,
        num_hidden_layers=model_config.num_hidden_layers,
        num_attention_heads=model_config.num_attention_heads,
        num_key_value_heads=model_config.num_key_value_heads,
        max_position_embeddings=model_config.max_position_embeddings,
        rms_norm_eps=model_config.rms_norm_eps,
        rope_theta=model_config.rope_theta,
        attention_bias=model_config.attention_bias,
    )

    # Initialize a Hugging Face model with the same configuration
    hf_model = HFLlamaForCausalLM(config=hf_config)

    # Remove sharding and convert JAX arrays to NumPy arrays
    model_params, _ = eqx.partition(model, eqx.is_array)
    model_params = jax.tree_util.tree_map(lambda x: np.array(x), model_params)

    # Copy weights from Equinox model to Hugging Face model
    # Embedding weights
    hf_model.model.embed_tokens.weight.data = torch.tensor(
        model_params.model.embed_tokens.weight, dtype=torch.float32
    )
    hf_model.lm_head.weight.data = torch.tensor(
        model_params.lm_head.weight, dtype=torch.float32
    )
    hf_model.model.norm.weight.data = torch.tensor(
        model_params.model.norm.weight, dtype=torch.float32
    )

    # Layer-wise weights
    for i in range(len(hf_model.model.layers)):
        eqx_layer = model_params.model.layers[i]
        hf_layer = hf_model.model.layers[i]

        # Self-attention weights
        hf_layer.self_attn.q_proj.weight.data = torch.tensor(
            eqx_layer.self_attn.q_proj.weight, dtype=torch.float32
        )
        hf_layer.self_attn.k_proj.weight.data = torch.tensor(
            eqx_layer.self_attn.k_proj.weight, dtype=torch.float32
        )
        hf_layer.self_attn.v_proj.weight.data = torch.tensor(
            eqx_layer.self_attn.v_proj.weight, dtype=torch.float32
        )
        hf_layer.self_attn.o_proj.weight.data = torch.tensor(
            eqx_layer.self_attn.o_proj.weight, dtype=torch.float32
        )

        # MLP weights
        hf_layer.mlp.gate_proj.weight.data = torch.tensor(
            eqx_layer.mlp.gate_proj.weight, dtype=torch.float32
        )
        hf_layer.mlp.up_proj.weight.data = torch.tensor(
            eqx_layer.mlp.up_proj.weight, dtype=torch.float32
        )
        hf_layer.mlp.down_proj.weight.data = torch.tensor(
            eqx_layer.mlp.down_proj.weight, dtype=torch.float32
        )

        # Layer norms
        hf_layer.input_layernorm.weight.data = torch.tensor(
            eqx_layer.input_layernorm.weight, dtype=torch.float32
        )
        hf_layer.post_attention_layernorm.weight.data = torch.tensor(
            eqx_layer.post_attention_layernorm.weight, dtype=torch.float32
        )

    # Save the Hugging Face model
    hf_model.save_pretrained(output_dir)

    # Save the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.save_pretrained(output_dir)

    print(f"Model and tokenizer saved to {output_dir}")
