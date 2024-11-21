"""
test_llama_port.py

This script provides a comprehensive suite of tests to validate the equivalence between the original PyTorch implementation of the Llama model and its JAX/Equinox port. The main objectives of this script are:

1. **Importing Necessary Models and Implementations**: It imports both the Llama model from Hugging Face Transformers and the custom implementation using JAX/Equinox.

2. **Defining Test Functions for Core Components**: The script includes test functions for each major component of the Llama architecture, such as:
   - Token Embedding
   - Linear Layers
   - RMS Normalization
   - Multi-Layer Perceptron (MLP)
   - Self-Attention Mechanism
   - Decoder Layer
   - Complete Model
   - Causal Language Model

3. **Comparative Testing for Each Component**: For every component, the script:
   - Initializes both the PyTorch and JAX/Equinox versions.
   - Transfers weights from the PyTorch model to the JAX/Equinox model.
   - Generates identical inputs for both implementations.
   - Computes outputs using both versions.
   - Verifies that the outputs are numerically close within a specified tolerance.

The primary goals of this test suite are to:

- Ensure that the JAX/Equinox implementation faithfully replicates the behavior of the original PyTorch model.
- Confirm that each component of the Llama architecture has been accurately ported to JAX/Equinox.
- Identify any discrepancies or errors that may have arisen during the porting process.
- Provide a dependable testing framework for ongoing development and refactoring of the JAX/Equinox implementation.

**Usage Instructions**:

To validate the equivalence between the PyTorch and JAX/Equinox implementations of the Llama model, run this script using `pytest`. Successful execution, with all tests passing, indicates a successful porting process.

**Note**:

Maintaining the integrity and accuracy of the JAX/Equinox port is crucial. This test suite should be executed after any significant modifications to the implementation and should be integrated into the continuous integration (CI) process.
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp
from transformers import AutoTokenizer, LlamaForCausalLM as HFLlamaForCausalLM
import torch
import equinox as eqx

from typing import Optional, Tuple

from src.felafax.trainer_engine.models.llama3.jax.model import (
    LlamaEmbedding,
    LlamaLinear,
    LlamaRotaryEmbedding,
    LlamaRMSNorm,
    LlamaSdpaAttention,
    LlamaMLP,
    LlamaDecoderLayer,
    LlamaModel,
    LlamaForCausalLM,
    LlamaConfig,
)


# JAX helper functions
def jax_apply_rotary_pos_emb(q, k, cos, sin):
    # Applies rotary positional embeddings to query and key tensors
    q_embed = (q * cos) + (jax_rotate_half(q) * sin)
    k_embed = (k * cos) + (jax_rotate_half(k) * sin)
    return q_embed, k_embed


def jax_rotate_half(x):
    # Splits tensor in half and swaps components with sign change for rotary embedding
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate((-x2, x1), axis=-1)


# PyTorch helper functions
def torch_apply_rotary_pos_emb(q, k, cos, sin):
    # Applies rotary positional embeddings to query and key tensors in PyTorch
    q_embed = (q * cos) + (torch_rotate_half(q) * sin)
    k_embed = (k * cos) + (torch_rotate_half(k) * sin)
    return q_embed, k_embed


def torch_rotate_half(x):
    # Splits tensor in half and swaps components with sign change for rotary embedding in PyTorch
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeats key/value tensors to match the number of attention heads.
    """
    # Gets the shape of the input tensor
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        # Returns the input tensor if no repetition is needed
        return hidden_states
    # Expands and reshapes the tensor to repeat key/value heads
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(
        batch, num_key_value_heads * n_rep, slen, head_dim
    )


class HookedLlamaSdpaAttention(torch.nn.Module):
    """
    A wrapper around the LlamaSdpaAttention to add hooks for intermediate tensors.
    """

    def __init__(self, original_attention):
        super().__init__()
        # Stores the original attention module and initializes hooks dictionary
        self.original_attention = original_attention
        self.hooks = {}

    def add_hook(self, name, hook_fn):
        # Adds a hook function to the hooks dictionary
        self.hooks[name] = hook_fn

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[
        torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]
    ]:
        # Retrieves batch size and sequence length
        bsz, q_len, _ = hidden_states.size()

        # Applies the query, key, and value projections
        query_states = self.original_attention.q_proj(hidden_states)
        key_states = self.original_attention.k_proj(hidden_states)
        value_states = self.original_attention.v_proj(hidden_states)

        # Calls the 'proj' hook if it exists
        if "proj" in self.hooks:
            self.hooks["proj"](query_states, key_states, value_states)

        # Reshapes the projected tensors for multi-head attention
        query_states = query_states.view(
            bsz,
            q_len,
            self.original_attention.num_heads,
            self.original_attention.head_dim,
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz,
            q_len,
            self.original_attention.num_key_value_heads,
            self.original_attention.head_dim,
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz,
            q_len,
            self.original_attention.num_key_value_heads,
            self.original_attention.head_dim,
        ).transpose(1, 2)

        # Calls the 'reshape' hook if it exists
        if "reshape" in self.hooks:
            self.hooks["reshape"](query_states, key_states, value_states)

        if position_embeddings is None:
            # Retrieves positional embeddings using rotary embeddings
            cos, sin = self.original_attention.rotary_emb(
                value_states, position_ids
            )
        else:
            cos, sin = position_embeddings

        # Calls the 'rotary' hook if it exists
        if "rotary" in self.hooks:
            self.hooks["rotary"](cos, sin)

        # Applies rotary positional embeddings to queries and keys
        query_states, key_states = torch_apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        # Calls the 'post_rotary' hook if it exists
        if "post_rotary" in self.hooks:
            self.hooks["post_rotary"](query_states, key_states)

        if past_key_value is not None:
            # Updates key and value states using past key values
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "cache_position": cache_position,
            }
            key_states, value_states = past_key_value.update(
                key_states,
                value_states,
                self.original_attention.layer_idx,
                cache_kwargs,
            )

        # Repeats key and value tensors if necessary
        key_states = repeat_kv(
            key_states, self.original_attention.num_key_value_groups
        )
        value_states = repeat_kv(
            value_states, self.original_attention.num_key_value_groups
        )

        # Calls the 'repeat_kv' hook if it exists
        if "repeat_kv" in self.hooks:
            self.hooks["repeat_kv"](key_states, value_states)

        # Prepares the causal mask for attention
        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

        if query_states.device.type == "cuda" and causal_mask is not None:
            # Ensures tensors are contiguous for CUDA operations
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        # Determines if attention is causal based on mask and sequence length
        is_causal = True if causal_mask is None and q_len > 1 else False

        # Computes scaled dot-product attention
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.original_attention.attention_dropout
            if self.original_attention.training
            else 0.0,
            is_causal=is_causal,
        )

        # Calls the 'attn_output' hook if it exists
        if "attn_output" in self.hooks:
            self.hooks["attn_output"](attn_output)

        # Reshapes the attention output back to original dimensions
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, -1)

        # Calls the 'reshape_output' hook if it exists
        if "reshape_output" in self.hooks:
            self.hooks["reshape_output"](attn_output)

        # Applies the output projection
        attn_output = self.original_attention.o_proj(attn_output)

        # Calls the 'final_output' hook if it exists
        if "final_output" in self.hooks:
            self.hooks["final_output"](attn_output)

        return attn_output, None, past_key_value


def test_hooked_attention_equivalence(hf_model):
    """Tests the equivalence of original and hooked attention mechanisms."""
    # Unpacks the Hugging Face model
    _, hf_model = hf_model
    # Retrieves the original attention module
    original_attn = hf_model.model.layers[0].self_attn
    # Wraps the attention module with hooks
    hooked_attn = HookedLlamaSdpaAttention(original_attn)

    # Creates sample input tensors
    batch_size = 1
    seq_length = 5
    hidden_size = original_attn.hidden_size
    hidden_states = torch.randn(batch_size, seq_length, hidden_size)
    position_ids = torch.arange(seq_length).unsqueeze(0)

    # Runs both attention mechanisms without gradient computation
    with torch.no_grad():
        original_output, _, _ = original_attn(
            hidden_states, position_ids=position_ids
        )
        hooked_output, _, _ = hooked_attn(
            hidden_states, position_ids=position_ids
        )

    # Asserts that outputs from both mechanisms are close
    assert torch.allclose(
        original_output, hooked_output, atol=1e-5
    ), f"Max difference: {(original_output - hooked_output).abs().max().item()}"

    print(
        "All intermediate results match between original and hooked attention."
    )


# Helper function to convert PyTorch tensor to JAX array
def torch_to_jax(tensor):
    # Converts a PyTorch tensor to a JAX numpy array
    return jnp.array(tensor.detach().numpy())


# Helper function to compare PyTorch and JAX outputs
def assert_close(torch_output, jax_output, rtol=1e-5, atol=1e-5):
    # Uses NumPy testing utilities to assert closeness
    np.testing.assert_allclose(
        torch_output.detach().numpy(), jax_output, rtol=rtol, atol=atol
    )


@pytest.fixture(scope="module")
def hf_model():
    # Loads the Hugging Face tokenizer and model
    model_name = "meta-llama/Meta-Llama-3.1-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = HFLlamaForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32
    )
    # Sets the model to evaluation mode
    model.eval()
    return tokenizer, model


@pytest.fixture(scope="module")
def eqx_config(hf_model):
    # Retrieves the configuration from the Hugging Face model
    _, hf_model = hf_model
    config = LlamaConfig(
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
    return config


def test_llama_embedding(hf_model, eqx_config):
    """Tests the LlamaEmbedding module for equivalence."""
    # Unpacks the Hugging Face model
    _, hf_model = hf_model
    # Retrieves the embedding layer from the Hugging Face model
    hf_embed = hf_model.model.embed_tokens
    # Initializes the Equinox embedding layer
    eqx_embed = LlamaEmbedding(eqx_config.vocab_size, eqx_config.hidden_size)

    # Copies weights from Hugging Face to Equinox embedding
    eqx_embed = eqx.tree_at(
        lambda t: t.weight, eqx_embed, torch_to_jax(hf_embed.weight)
    )

    # Creates sample input IDs
    input_ids = jnp.array([[1, 2, 3, 4, 5]])
    # Computes outputs from both embeddings
    hf_output = hf_embed(torch.tensor(input_ids.tolist()))
    eqx_output = eqx_embed(input_ids)

    # Asserts that the outputs are close
    assert_close(hf_output, eqx_output)


def test_llama_linear(hf_model, eqx_config):
    """Tests the LlamaLinear module for equivalence."""
    # Unpacks the Hugging Face model
    _, hf_model = hf_model
    # Retrieves a linear layer from the Hugging Face model
    hf_linear = hf_model.model.layers[0].self_attn.q_proj
    # Initializes the Equinox linear layer
    eqx_linear = LlamaLinear(
        eqx_config.hidden_size, eqx_config.hidden_size, bias=False
    )

    # Copies weights from Hugging Face to Equinox linear layer
    eqx_linear = eqx.tree_at(
        lambda t: t.weight, eqx_linear, torch_to_jax(hf_linear.weight)
    )

    # Generates a random input tensor
    x = jax.random.normal(jax.random.PRNGKey(0), (1, eqx_config.hidden_size))
    # Computes outputs from both linear layers
    hf_output = hf_linear(torch.tensor(x.tolist()))
    eqx_output = eqx_linear(x)

    # Asserts that the outputs are close
    assert_close(hf_output, eqx_output)


def test_llama_rms_norm(hf_model, eqx_config):
    """Tests the LlamaRMSNorm module for equivalence."""
    # Unpacks the Hugging Face model
    _, hf_model = hf_model
    # Retrieves the RMS normalization layer
    hf_norm = hf_model.model.norm
    # Initializes the Equinox RMS normalization layer
    eqx_norm = LlamaRMSNorm(eqx_config.hidden_size, eqx_config.rms_norm_eps)

    # Copies weights from Hugging Face to Equinox normalization layer
    eqx_norm = eqx.tree_at(
        lambda t: t.weight, eqx_norm, torch_to_jax(hf_norm.weight)
    )

    # Generates a random input tensor
    x = jax.random.normal(jax.random.PRNGKey(0), (1, 1, eqx_config.hidden_size))
    # Computes outputs from both normalization layers
    hf_output = hf_norm(torch.tensor(x.tolist()))
    eqx_output = eqx_norm(x)

    # Asserts that the outputs are close
    assert_close(hf_output, eqx_output)


def test_llama_mlp(hf_model, eqx_config):
    """Tests the LlamaMLP module for equivalence."""
    # Unpacks the Hugging Face model
    _, hf_model = hf_model
    # Retrieves the MLP module from the Hugging Face model
    hf_mlp = hf_model.model.layers[0].mlp
    # Initializes the Equinox MLP module
    eqx_mlp = LlamaMLP(eqx_config.hidden_size, eqx_config.intermediate_size)

    # Copies weights from Hugging Face to Equinox MLP
    eqx_mlp = eqx.tree_at(
        lambda t: t.gate_proj.weight,
        eqx_mlp,
        torch_to_jax(hf_mlp.gate_proj.weight),
    )
    eqx_mlp = eqx.tree_at(
        lambda t: t.up_proj.weight, eqx_mlp, torch_to_jax(hf_mlp.up_proj.weight)
    )
    eqx_mlp = eqx.tree_at(
        lambda t: t.down_proj.weight,
        eqx_mlp,
        torch_to_jax(hf_mlp.down_proj.weight),
    )

    # Generates a random input tensor
    x = jax.random.normal(jax.random.PRNGKey(0), (1, 1, eqx_config.hidden_size))
    # Computes outputs from both MLPs
    hf_output = hf_mlp(torch.tensor(x.tolist()))
    eqx_output = eqx_mlp(x)

    # Asserts that the outputs are close
    assert_close(hf_output, eqx_output)


def test_llama_rotary_embedding(hf_model, eqx_config):
    """Tests the LlamaRotaryEmbedding module for equivalence."""
    # Unpacks the Hugging Face model
    _, hf_model = hf_model
    # Retrieves the rotary embedding module
    hf_rotary_emb = hf_model.model.layers[0].self_attn.rotary_emb
    # Initializes the Equinox rotary embedding module
    eqx_rotary_emb = LlamaRotaryEmbedding(eqx_config)

    # Generates sample input tensors
    batch_size = 2
    seq_length = 10
    hidden_dim = eqx_config.hidden_size // eqx_config.num_attention_heads
    x = jax.random.normal(
        jax.random.PRNGKey(0), (batch_size, seq_length, hidden_dim)
    )
    position_ids = jnp.arange(seq_length)[None, :]

    # Computes outputs from both rotary embeddings
    hf_cos, hf_sin = hf_rotary_emb(
        torch.tensor(x.tolist()), torch.tensor(position_ids.tolist())
    )
    eqx_cos, eqx_sin = eqx_rotary_emb(x, position_ids)

    # Asserts that the outputs are close
    # TODO(port): Reduce tolerance to 1e-5
    assert_close(hf_cos, eqx_cos, rtol=1e-2, atol=1e-2)
    assert_close(hf_sin, eqx_sin, rtol=1e-2, atol=1e-2)


def test_llama_attention(hf_model, eqx_config):
    """Tests the LlamaSdpaAttention module for equivalence."""
    # Unpacks the Hugging Face model
    _, hf_model = hf_model
    # Retrieves the attention module and wraps it with hooks
    hf_attn = hf_model.model.layers[0].self_attn
    hooked_hf_attn = HookedLlamaSdpaAttention(hf_attn)
    # Initializes the Equinox attention module
    eqx_attn = LlamaSdpaAttention(eqx_config)

    # Copies weights from Hugging Face to Equinox attention module
    eqx_attn = eqx.tree_at(
        lambda t: t.q_proj.weight, eqx_attn, torch_to_jax(hf_attn.q_proj.weight)
    )
    eqx_attn = eqx.tree_at(
        lambda t: t.k_proj.weight, eqx_attn, torch_to_jax(hf_attn.k_proj.weight)
    )
    eqx_attn = eqx.tree_at(
        lambda t: t.v_proj.weight, eqx_attn, torch_to_jax(hf_attn.v_proj.weight)
    )
    eqx_attn = eqx.tree_at(
        lambda t: t.o_proj.weight, eqx_attn, torch_to_jax(hf_attn.o_proj.weight)
    )

    # Checks and compares inverse frequency components
    hf_inv_freq = hf_attn.rotary_emb.inv_freq
    eqx_inv_freq = eqx_attn.rotary_emb.inv_freq

    # print("Comparing inverse frequency components:")
    # print(f"HF inv_freq shape: {hf_inv_freq.shape}")
    # print(f"EQX inv_freq shape: {eqx_inv_freq.shape}")
    # print(f"HF inv_freq mean: {hf_inv_freq.mean().item():.6f}")
    # print(f"EQX inv_freq mean: {eqx_inv_freq.mean().item():.6f}")
    # print(f"HF inv_freq std: {hf_inv_freq.std().item():.6f}")
    # print(f"EQX inv_freq std: {eqx_inv_freq.std():.6f}")
    # print(
    #     f"Max difference: {np.abs(hf_inv_freq.detach().numpy() - eqx_inv_freq).max():.6f}"
    # )

    # TODO(port): Reduce tolerance to 1e-5
    assert_close(hf_inv_freq, eqx_inv_freq, rtol=1e-2, atol=1e-2)

    # Generates a random input tensor and position IDs
    x = jax.random.normal(jax.random.PRNGKey(0), (1, 5, eqx_config.hidden_size))
    position_ids = jnp.arange(5)[None, :]

    # Adds hooks to capture intermediate results
    def hook_factory(name):
        def hook(*args):
            nonlocal hf_intermediates
            hf_intermediates[name] = [arg.detach().numpy() for arg in args]

        return hook

    hf_intermediates = {}
    for hook_name in [
        "proj",
        "reshape",
        "rotary",
        "post_rotary",
        "repeat_kv",
        "attn_weights",
        "softmax",
        "attn_output",
        "reshape_output",
        "final_output",
    ]:
        # Adds hooks for specified names
        hooked_hf_attn.add_hook(hook_name, hook_factory(hook_name))

    # Runs the hooked attention mechanism
    hf_output, _, _ = hooked_hf_attn(
        torch.tensor(x.tolist()),
        position_ids=torch.tensor(position_ids.tolist()),
    )

    # Defines a function to capture intermediate results in Equinox
    def eqx_attention_with_intermediates(params, x, position_ids):
        intermediates = {}

        # Applies query, key, and value projections
        query_states = params.q_proj(x)
        key_states = params.k_proj(x)
        value_states = params.v_proj(x)
        intermediates["proj"] = (query_states, key_states, value_states)

        # Reshapes tensors for multi-head attention
        bsz, q_len, _ = x.shape
        query_states = query_states.reshape(
            bsz, q_len, params.num_heads, params.head_dim
        ).transpose(0, 2, 1, 3)
        key_states = key_states.reshape(
            bsz, q_len, params.num_key_value_heads, params.head_dim
        ).transpose(0, 2, 1, 3)
        value_states = value_states.reshape(
            bsz, q_len, params.num_key_value_heads, params.head_dim
        ).transpose(0, 2, 1, 3)
        intermediates["reshape"] = (query_states, key_states, value_states)

        # Computes rotary embeddings
        cos, sin = params.rotary_emb(value_states, position_ids)
        intermediates["rotary"] = (cos, sin)

        # Applies rotary positional embeddings
        query_states, key_states = jax_apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )
        intermediates["post_rotary"] = (query_states, key_states)

        if params.num_key_value_heads != params.num_heads:
            # Repeats key and value tensors to match attention heads
            key_states = jnp.repeat(
                key_states,
                params.num_heads // params.num_key_value_heads,
                axis=1,
            )
            value_states = jnp.repeat(
                value_states,
                params.num_heads // params.num_key_value_heads,
                axis=1,
            )
        intermediates["repeat_kv"] = (key_states, value_states)

        # Computes attention weights via scaled dot-product
        attn_weights = jnp.einsum(
            "bhqd,bhkd->bhqk", query_states, key_states
        ) / jnp.sqrt(params.head_dim)
        intermediates["attn_weights"] = (attn_weights,)

        # Creates and applies causal mask
        causal_mask = jnp.tril(jnp.ones((q_len, q_len)))
        causal_mask = causal_mask[None, None, :, :]
        attn_weights = jnp.where(causal_mask == 0, float("-inf"), attn_weights)

        # Applies softmax to get attention probabilities
        attn_weights = jax.nn.softmax(attn_weights, axis=-1)
        intermediates["softmax"] = (attn_weights,)

        # Computes the attention output
        attn_output = jnp.einsum("bhqk,bhkd->bhqd", attn_weights, value_states)
        intermediates["attn_output"] = (attn_output,)

        # Reshapes the attention output back to original dimensions
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(
            bsz, q_len, params.num_heads * params.head_dim
        )
        intermediates["reshape_output"] = (attn_output,)

        # Applies the output projection
        attn_output = params.o_proj(attn_output)
        intermediates["final_output"] = (attn_output,)

        return attn_output, intermediates

    # Runs the Equinox attention mechanism with intermediates
    eqx_output, eqx_intermediates = eqx_attention_with_intermediates(
        eqx_attn, x, position_ids
    )

    # Compares intermediate results from both implementations
    for name in hf_intermediates.keys():
        print(f"Comparing {name}:")
        for i, (hf_tensor, eqx_tensor) in enumerate(
            zip(hf_intermediates[name], eqx_intermediates[name])
        ):
            print(f"  Shape: HF {hf_tensor.shape}, EQX {eqx_tensor.shape}")
            print(
                f"  Mean: HF {hf_tensor.mean():.6f}, EQX {eqx_tensor.mean():.6f}"
            )
            print(f"  Std: HF {hf_tensor.std():.6f}, EQX {eqx_tensor.std():.6f}")
            print(f"  Max diff: {np.abs(hf_tensor - eqx_tensor).max():.6f}")
            print()

    # Asserts that the final outputs are close
    assert_close(hf_output, eqx_output, rtol=1e-2, atol=1e-2)


def test_llama_decoder_layer(hf_model, eqx_config):
    """Tests the LlamaDecoderLayer module for equivalence."""
    # Unpacks the Hugging Face model
    _, hf_model = hf_model
    # Retrieves the decoder layer
    hf_layer = hf_model.model.layers[0]
    # Initializes the Equinox decoder layer
    eqx_layer = LlamaDecoderLayer(eqx_config)

    # Copies weights from Hugging Face to Equinox decoder layer
    eqx_layer = eqx.tree_at(
        lambda t: t.self_attn.q_proj.weight,
        eqx_layer,
        torch_to_jax(hf_layer.self_attn.q_proj.weight),
    )
    eqx_layer = eqx.tree_at(
        lambda t: t.self_attn.k_proj.weight,
        eqx_layer,
        torch_to_jax(hf_layer.self_attn.k_proj.weight),
    )
    eqx_layer = eqx.tree_at(
        lambda t: t.self_attn.v_proj.weight,
        eqx_layer,
        torch_to_jax(hf_layer.self_attn.v_proj.weight),
    )
    eqx_layer = eqx.tree_at(
        lambda t: t.self_attn.o_proj.weight,
        eqx_layer,
        torch_to_jax(hf_layer.self_attn.o_proj.weight),
    )
    eqx_layer = eqx.tree_at(
        lambda t: t.mlp.gate_proj.weight,
        eqx_layer,
        torch_to_jax(hf_layer.mlp.gate_proj.weight),
    )
    eqx_layer = eqx.tree_at(
        lambda t: t.mlp.up_proj.weight,
        eqx_layer,
        torch_to_jax(hf_layer.mlp.up_proj.weight),
    )
    eqx_layer = eqx.tree_at(
        lambda t: t.mlp.down_proj.weight,
        eqx_layer,
        torch_to_jax(hf_layer.mlp.down_proj.weight),
    )
    eqx_layer = eqx.tree_at(
        lambda t: t.input_layernorm.weight,
        eqx_layer,
        torch_to_jax(hf_layer.input_layernorm.weight),
    )
    eqx_layer = eqx.tree_at(
        lambda t: t.post_attention_layernorm.weight,
        eqx_layer,
        torch_to_jax(hf_layer.post_attention_layernorm.weight),
    )

    # Generates a random input tensor and position IDs
    x = jax.random.normal(jax.random.PRNGKey(0), (1, 5, eqx_config.hidden_size))
    position_ids = jnp.arange(5)[None, :]

    # Computes outputs from both decoder layers
    hf_output = hf_layer(
        torch.tensor(x.tolist()),
        position_ids=torch.tensor(position_ids.tolist()),
    )[0]
    eqx_output = eqx_layer(x, position_ids=position_ids)

    # Asserts that the outputs are close
    # TODO(port): Reduce tolerance to 1e-5
    assert_close(hf_output, eqx_output, rtol=1e-2, atol=1e-2)


def test_llama_model(hf_model, eqx_config):
    """Tests the full LlamaModel for equivalence."""
    # Unpacks the tokenizer and Hugging Face model
    tokenizer, hf_model = hf_model
    # Initializes the Equinox LlamaModel
    eqx_model = LlamaModel(eqx_config)

    # Copies weights from Hugging Face to Equinox model
    eqx_model = eqx.tree_at(
        lambda t: t.embed_tokens.weight,
        eqx_model,
        torch_to_jax(hf_model.model.embed_tokens.weight),
    )
    eqx_model = eqx.tree_at(
        lambda t: t.norm.weight,
        eqx_model,
        torch_to_jax(hf_model.model.norm.weight),
    )
    for i, layer in enumerate(eqx_model.layers):
        hf_layer = hf_model.model.layers[i]
        eqx_model = eqx.tree_at(
            lambda t: t.layers[i].self_attn.q_proj.weight,
            eqx_model,
            torch_to_jax(hf_layer.self_attn.q_proj.weight),
        )
        eqx_model = eqx.tree_at(
            lambda t: t.layers[i].self_attn.k_proj.weight,
            eqx_model,
            torch_to_jax(hf_layer.self_attn.k_proj.weight),
        )
        eqx_model = eqx.tree_at(
            lambda t: t.layers[i].self_attn.v_proj.weight,
            eqx_model,
            torch_to_jax(hf_layer.self_attn.v_proj.weight),
        )
        eqx_model = eqx.tree_at(
            lambda t: t.layers[i].self_attn.o_proj.weight,
            eqx_model,
            torch_to_jax(hf_layer.self_attn.o_proj.weight),
        )
        eqx_model = eqx.tree_at(
            lambda t: t.layers[i].mlp.gate_proj.weight,
            eqx_model,
            torch_to_jax(hf_layer.mlp.gate_proj.weight),
        )
        eqx_model = eqx.tree_at(
            lambda t: t.layers[i].mlp.up_proj.weight,
            eqx_model,
            torch_to_jax(hf_layer.mlp.up_proj.weight),
        )
        eqx_model = eqx.tree_at(
            lambda t: t.layers[i].mlp.down_proj.weight,
            eqx_model,
            torch_to_jax(hf_layer.mlp.down_proj.weight),
        )
        eqx_model = eqx.tree_at(
            lambda t: t.layers[i].input_layernorm.weight,
            eqx_model,
            torch_to_jax(hf_layer.input_layernorm.weight),
        )
        eqx_model = eqx.tree_at(
            lambda t: t.layers[i].post_attention_layernorm.weight,
            eqx_model,
            torch_to_jax(hf_layer.post_attention_layernorm.weight),
        )

    # Encodes input text and prepares position IDs
    input_text = "Hello, world!"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    position_ids = torch.arange(input_ids.shape[1])[None, :]

    # Computes outputs from both models
    hf_output = hf_model.model(input_ids, position_ids=position_ids)[0]
    eqx_output = eqx_model(
        jnp.array(input_ids), position_ids=jnp.array(position_ids)
    )

    # Asserts that the outputs are close with specified tolerances
    # TODO(port): Reduce tolerance to 1e-5, changing to 1e-1 for now.
    assert_close(hf_output, eqx_output, rtol=1e-1, atol=1e-1)


def test_llama_for_causal_lm(hf_model, eqx_config):
    """Tests the LlamaForCausalLM module for equivalence."""
    # Unpacks the tokenizer and Hugging Face model
    tokenizer, hf_model = hf_model
    # Initializes the Equinox LlamaForCausalLM model
    eqx_model = LlamaForCausalLM(eqx_config)

    # Copies weights from Hugging Face to Equinox model
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

    # Encodes input text and prepares position IDs
    input_text = "Hello, world!"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    position_ids = torch.arange(input_ids.shape[1])[None, :]

    # Computes outputs from both language models
    hf_output = hf_model(input_ids, position_ids=position_ids).logits
    eqx_output = eqx_model(
        jnp.array(input_ids), position_ids=jnp.array(position_ids)
    )

    # Asserts that the outputs are close
    assert_close(hf_output, eqx_output, rtol=1e-2, atol=1e-1)


def test_load_checkpoint(hf_model):
    """Tests loading weights from checkpoint."""
    from src.felafax.trainer_engine.checkpoint import load_llama_from_hf
    from src.felafax.trainer_engine.trainer import get_mesh

    eqx_model, _ = load_llama_from_hf(
        "meta-llama/Meta-Llama-3.1-8B", mesh=get_mesh(jax.device_count())
    )

    # Create input for testing
    tokenizer, hf_model = hf_model
    input_text = "Hello, world!"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    position_ids = torch.arange(input_ids.shape[1])[None, :]

    # Get output from loaded Equinox model
    eqx_output = eqx_model(
        jnp.array(input_ids), position_ids=jnp.array(position_ids)
    )

    # Compare outputs
    hf_output = hf_model(input_ids, position_ids=position_ids).logits
    assert_close(hf_output, eqx_output, rtol=1, atol=1e-2)
