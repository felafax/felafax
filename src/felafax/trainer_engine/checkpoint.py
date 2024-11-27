"""Loads and saves checkpoint."""

import os
import json
from typing import Optional, Tuple, Any
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import torch
import equinox as eqx
import orbax.checkpoint as ocp
from transformers import (
    LlamaForCausalLM as HFLlamaForCausalLM,
    LlamaConfig as HFLlamaConfig,
    AutoTokenizer,
)
from src.felafax.trainer_engine.models.llama3.jax.model import (
    LlamaConfig,
    LlamaForCausalLM,
    LlamaLinear,
)

from typing import Optional, Tuple, Any
from jaxtyping import PyTree
from jax.sharding import NamedSharding, PartitionSpec as PS
from jax.experimental import mesh_utils

from .models.llama3.jax.model import (
    LlamaConfig,
    LlamaForCausalLM,
)
from .utils import named_tree_map


@dataclass
class CheckpointerConfig:
    """Configuration for checkpointing"""

    checkpoint_dir: str
    max_to_keep: int = 2
    save_interval_steps: int = 10
    create: bool = True  # Create the checkpoint directory if it doesn't exist
    enable_async_checkpointing: bool = True
    erase_existing_checkpoints: bool = False


class Checkpointer:
    def __init__(self, config: CheckpointerConfig):
        if not config.checkpoint_dir:
            raise ValueError("Checkpoint directory cannot be empty")
        self.config = config
        self.checkpoint_dir = config.checkpoint_dir
        if config.erase_existing_checkpoints:
            ocp.test_utils.erase_and_create_empty(self.checkpoint_dir)

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
            force=True,
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
    model_name: str, mesh: jax.sharding.Mesh, token: Optional[str] = None
):
    """Loads a model from a checkpoint or Hugging Face.

    Args:
        model_name: Name or path of the model to load
        token: HuggingFace token for accessing gated models
    """
    return load_llama_from_hf(model_name, mesh=mesh, token=token)


def load_checkpoint_or_model(
    model_name: str,
    mesh: jax.sharding.Mesh,
    checkpointer: Checkpointer,
    param_dtype=jnp.float32,
    compute_dtype=jnp.float32,
) -> LlamaForCausalLM:
    """Loads checkpoint from local storage using Orbax or downloads from HF with specified dtypes.

    Args:
        model_name: Name of HF model (e.g. 'meta-llama/Llama-2-7b') or path to local checkpoint
        checkpointer: An instance of Checkpointer to manage loading
        param_dtype: The dtype in which parameters are stored and loaded
        compute_dtype: The dtype in which computations are performed and outputs are returned

    Returns:
        tuple: (model, model_config)
    """
    has_checkpoints = len(checkpointer.checkpoint_mgr.all_steps()) > 0
    if has_checkpoints:
        # Restores the model in whatever dtypes are stored in the checkpoint.
        model, model_config = checkpointer.restore_checkpoint()
        print(
            f"Restored checkpoint from step {checkpointer.checkpoint_mgr.latest_step()}"
        )
        return model, model_config

    model, model_config = load_llama_from_hf(
        model_name,
        mesh=mesh,
        param_dtype=param_dtype,
        compute_dtype=compute_dtype,
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


def _make_torch_to_jax(dtype, mesh):
    """Creates a closure that converts PyTorch tensors to JAX arrays with sharding annotations."""

    def _torch_to_jax(tensor, sharding_spec):
        jax_array = jnp.array(tensor.detach().numpy(), dtype=dtype)
        sharding = NamedSharding(mesh, sharding_spec)
        return jax.device_put(jax_array, sharding)

    return _torch_to_jax


# TODO(refactor): Move load model into models/llama.
def load_llama_from_hf(
    model_name: str,
    mesh: jax.sharding.Mesh,
    token: Optional[str] = None,
    lora_rank: int = 0,
    param_dtype: Any = jnp.float32,
    compute_dtype: Any = jnp.float32,
) -> LlamaForCausalLM:
    """Downloads and converts Hugging Face model to Equinox model with specified dtypes.

    Args:
        model_name: Name of the Hugging Face model to load
        mesh: JAX sharding mesh
        token: HuggingFace token for accessing gated models
        lora_rank: Rank for LoRA parameters (set to 0 for no LoRA)
        param_dtype: The dtype in which parameters are stored
        compute_dtype: The dtype in which computations are performed

    Returns:
        eqx_model: LlamaForCausalLM model with specified dtypes
        model_config: Configuration of the model
    """
    # Load HF model on CPU to save GPU memory
    hf_model = HFLlamaForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        token=token,
        device_map={"": "cpu"},
    )

    # Create config and initialize Equinox model
    model_config = create_llama_config_from_hf_model(hf_model)
    model_config.lora_rank = lora_rank

    key = jax.random.PRNGKey(42)
    eqx_model = LlamaForCausalLM(
        model_config,
        param_dtype=param_dtype,
        compute_dtype=compute_dtype,
        key=key,
    )

    # Conversion functions
    torch_to_jax_float32 = _make_torch_to_jax(dtype=jnp.float32, mesh=mesh)
    torch_to_jax = _make_torch_to_jax(dtype=param_dtype, mesh=mesh)

    # Copy embedding and output layers
    eqx_model = eqx.tree_at(
        lambda m: m.model.embed_tokens.weight,
        eqx_model,
        torch_to_jax_float32(
            hf_model.model.embed_tokens.weight, PS(("mp", "fsdp"))
        ),
    )
    eqx_model = eqx.tree_at(
        lambda m: m.model.norm.weight,
        eqx_model,
        torch_to_jax_float32(hf_model.model.norm.weight, PS()),
    )
    eqx_model = eqx.tree_at(
        lambda m: m.lm_head.weight,
        eqx_model,
        torch_to_jax(hf_model.lm_head.weight, PS(("fsdp", "mp"))),
    )

    def _copy_weights(from_hf_layer_name, to_eqx_layer, partition_spec, dtype):
        """Copies weights from HF layer to JAX array.

        Since transformer layers are stacked using vmap in LlamaModel (creating a leading layer dimension), we create an empty JAX array and copy weights layer-by-layer to match this stacked structure."""
        weight_arr = jnp.empty(to_eqx_layer.shape, dtype=dtype)
        torch_to_jax_converter = _make_torch_to_jax(dtype=dtype, mesh=mesh)

        for i in range(hf_model.config.num_hidden_layers):
            layer_path = from_hf_layer_name.split(".")
            current = hf_model.model.layers[i]
            for attr in layer_path:
                current = getattr(current, attr)

            weight_arr = weight_arr.at[i].set(
                torch_to_jax_converter(current.weight, partition_spec)
            )
        return weight_arr

    # Self-attention weights
    eqx_model = eqx.tree_at(
        lambda m: m.model.layers.self_attn.q_proj.weight,
        eqx_model,
        _copy_weights(
            "self_attn.q_proj",  # copy from this HF layer name
            eqx_model.model.layers.self_attn.q_proj.weight,  # to eqx layer
            PS(("fsdp", "mp")),
            param_dtype,
        ),
    )

    eqx_model = eqx.tree_at(
        lambda m: m.model.layers.self_attn.k_proj.weight,
        eqx_model,
        _copy_weights(
            "self_attn.k_proj",
            eqx_model.model.layers.self_attn.k_proj.weight,
            PS(("fsdp", "mp")),
            param_dtype,
        ),
    )

    eqx_model = eqx.tree_at(
        lambda m: m.model.layers.self_attn.v_proj.weight,
        eqx_model,
        _copy_weights(
            "self_attn.v_proj",
            eqx_model.model.layers.self_attn.v_proj.weight,
            PS(("fsdp", "mp")),
            param_dtype,
        ),
    )

    eqx_model = eqx.tree_at(
        lambda m: m.model.layers.self_attn.o_proj.weight,
        eqx_model,
        _copy_weights(
            "self_attn.o_proj",
            eqx_model.model.layers.self_attn.o_proj.weight,
            PS(("mp", "fsdp")),
            param_dtype,
        ),
    )

    # MLP weights
    eqx_model = eqx.tree_at(
        lambda m: m.model.layers.mlp.gate_proj.weight,
        eqx_model,
        _copy_weights(
            "mlp.gate_proj",
            eqx_model.model.layers.mlp.gate_proj.weight,
            PS(("fsdp", "mp")),
            param_dtype,
        ),
    )

    eqx_model = eqx.tree_at(
        lambda m: m.model.layers.mlp.up_proj.weight,
        eqx_model,
        _copy_weights(
            "mlp.up_proj",
            eqx_model.model.layers.mlp.up_proj.weight,
            PS(("fsdp", "mp")),
            param_dtype,
        ),
    )

    eqx_model = eqx.tree_at(
        lambda m: m.model.layers.mlp.down_proj.weight,
        eqx_model,
        _copy_weights(
            "mlp.down_proj",
            eqx_model.model.layers.mlp.down_proj.weight,
            PS(("mp", "fsdp")),
            param_dtype,
        ),
    )

    # Layer norms (using float32)
    eqx_model = eqx.tree_at(
        lambda m: m.model.layers.input_layernorm.weight,
        eqx_model,
        _copy_weights(
            "input_layernorm",
            eqx_model.model.layers.input_layernorm.weight,
            PS(),
            jnp.float32,
        ),
    )

    eqx_model = eqx.tree_at(
        lambda m: m.model.layers.post_attention_layernorm.weight,
        eqx_model,
        _copy_weights(
            "post_attention_layernorm",
            eqx_model.model.layers.post_attention_layernorm.weight,
            PS(),
            jnp.float32,
        ),
    )

    return eqx_model, model_config


def save_model_to_hf(
    model: eqx.Module,
    model_config: LlamaConfig,
    output_dir: str,
    tokenizer_name: str = "meta-llama/Llama-2-7b-hf",
):
    """Converts and saves an Equinox model to Hugging Face format.

    Args:
        model: Equinox LlamaForCausalLM model instance.
        model_config: Corresponding model configuration.
        output_dir: Directory to save the Hugging Face model.
        tokenizer_name: Name of the tokenizer to save alongside the model.
    """
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
    model_params = jax.tree_util.tree_map(
        lambda x: np.array(x).astype(np.float32),
        model_params,
    )

    def jax_to_torch(x):
        """Convert JAX array to PyTorch tensor."""
        return torch.tensor(jax.device_get(x), dtype=torch.float32)

    # Copy embedding and output layer weights
    hf_model.model.embed_tokens.weight.data = jax_to_torch(
        model_params.model.embed_tokens.weight
    )
    hf_model.lm_head.weight.data = jax_to_torch(model_params.lm_head.weight)
    hf_model.model.norm.weight.data = jax_to_torch(
        model_params.model.norm.weight
    )

    hf_layers = hf_model.model.layers
    def _copy_weights(from_eqx_layer, to_hf_layer_name):
        """Copies weights from vmapped Equinox layers to Hugging Face layers."""
        for i in range(len(hf_layers)):
            # Navigate through nested attributes to get the target layer (e.g. "self_attn.q_proj" -> layer.self_attn.q_proj)
            hf_submodule = hf_layers[i]
            for attr in to_hf_layer_name.split("."):
                hf_submodule = getattr(hf_submodule, attr)

            # Copy the weights from the eqx layer to hf submodule
            hf_submodule.weight.data = jax_to_torch(from_eqx_layer.weight[i])

    # Copy transformer layer weights
    _copy_weights(
        model_params.model.layers.self_attn.q_proj,
        "self_attn.q_proj",
    )
    _copy_weights(
        model_params.model.layers.self_attn.k_proj,
        "self_attn.k_proj",
    )
    _copy_weights(
        model_params.model.layers.self_attn.v_proj,
        "self_attn.v_proj",
    )
    _copy_weights(
        model_params.model.layers.self_attn.o_proj,
        "self_attn.o_proj",
    )

    # Copy MLP weights
    _copy_weights(
        model_params.model.layers.mlp.gate_proj,
        "mlp.gate_proj",
    )
    _copy_weights(
        model_params.model.layers.mlp.up_proj,
        "mlp.up_proj",
    )
    _copy_weights(
        model_params.model.layers.mlp.down_proj,
        "mlp.down_proj",
    )

    # Copy layer norm weights
    _copy_weights(
        model_params.model.layers.input_layernorm,
        "input_layernorm",
    )
    _copy_weights(
        model_params.model.layers.post_attention_layernorm,
        "post_attention_layernorm",
    )

    # Save model and tokenizer
    hf_model.save_pretrained(output_dir)

    # Save the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.save_pretrained(output_dir)

    print(f"Model and tokenizer saved to {output_dir}")
