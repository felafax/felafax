from typing import Any, Dict, Optional, Tuple
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.core import freeze, unfreeze
from jax.sharding import PartitionSpec as PS
from ....core.distributed import DistributedConfig
from ..config import LLaMAConfig


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    dim: int
    eps: float = 1e-6
    dtype: jnp.dtype = jnp.bfloat16

    def setup(self):
        self.weight = self.param("kernel", nn.initializers.ones, (self.dim,))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = x.astype(jnp.float32)
        variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        x = x * jax.lax.rsqrt(variance + self.eps)
        return (x * self.weight).astype(self.dtype)


class LoRALinear(nn.Module):
    """Linear layer with LoRA adaptation."""

    features: int
    lora_rank: int = 8
    lora_alpha: float = 16.0
    dtype: jnp.dtype = jnp.bfloat16
    use_bias: bool = False

    def setup(self):
        self.base_layer = nn.Dense(
            features=self.features,
            use_bias=self.use_bias,
            dtype=self.dtype,
            param_dtype=self.dtype,
        )

        # LoRA matrices
        self.lora_a = self.param(
            "lora_a",
            nn.initializers.normal(stddev=0.02),
            (self.base_layer.features, self.lora_rank),
        )
        self.lora_b = self.param(
            "lora_b", nn.initializers.zeros, (self.lora_rank, self.features)
        )

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        base_output = self.base_layer(inputs)
        if self.lora_rank > 0:
            lora_output = jnp.dot(jnp.dot(inputs, self.lora_a), self.lora_b)
            return base_output + (self.lora_alpha / self.lora_rank) * lora_output
        return base_output


def rotate_half(x: jnp.ndarray) -> jnp.ndarray:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return jnp.concatenate((-x2, x1), axis=-1)


def apply_rotary_pos_emb(
    q: jnp.ndarray, k: jnp.ndarray, cos: jnp.ndarray, sin: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Apply rotary position embeddings to query and key tensors."""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LLaMAAttention(nn.Module):
    """Multi-head attention with support for GQA and rotary embeddings."""

    config: LLaMAConfig
    dtype: jnp.dtype = jnp.bfloat16

    def setup(self):
        config = self.config
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads

        self.q_proj = LoRALinear(
            features=config.num_attention_heads * self.head_dim,
            dtype=self.dtype,
            lora_rank=config.lora_rank if config.use_lora else 0,
        )
        self.k_proj = LoRALinear(
            features=config.num_key_value_heads * self.head_dim,
            dtype=self.dtype,
            lora_rank=config.lora_rank if config.use_lora else 0,
        )
        self.v_proj = LoRALinear(
            features=config.num_key_value_heads * self.head_dim,
            dtype=self.dtype,
            lora_rank=config.lora_rank if config.use_lora else 0,
        )
        self.o_proj = LoRALinear(
            features=config.hidden_size,
            dtype=self.dtype,
            lora_rank=config.lora_rank if config.use_lora else 0,
        )

        self.attention_dropout = nn.Dropout(
            rate=config.attention_dropout if config.attention_dropout else 0.0
        )

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        batch_size, seq_length = hidden_states.shape[:2]

        # Project to query, key, value
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape
        query_states = query_states.reshape(
            batch_size, seq_length, self.num_heads, self.head_dim
        ).transpose(0, 2, 1, 3)
        key_states = key_states.reshape(
            batch_size, seq_length, self.num_kv_heads, self.head_dim
        ).transpose(0, 2, 1, 3)
        value_states = value_states.reshape(
            batch_size, seq_length, self.num_kv_heads, self.head_dim
        ).transpose(0, 2, 1, 3)

        # Compute attention weights
        attention_weights = jnp.matmul(
            query_states, key_states.transpose(0, 1, 3, 2)
        ) / jnp.sqrt(self.head_dim)

        # Apply attention mask if provided
        if attention_mask is not None:
            attention_weights = attention_weights + attention_mask

        attention_weights = jax.nn.softmax(attention_weights, axis=-1)

        # Apply attention dropout
        attention_weights = self.attention_dropout(
            attention_weights, deterministic=deterministic
        )

        # Compute attention output
        attn_output = jnp.matmul(attention_weights, value_states)
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(
            batch_size, seq_length, -1
        )

        return self.o_proj(attn_output)


class LLaMAMLP(nn.Module):
    """LLaMA MLP module."""

    config: LLaMAConfig
    dtype: jnp.dtype = jnp.bfloat16

    def setup(self):
        self.gate_proj = LoRALinear(
            features=self.config.intermediate_size,
            dtype=self.dtype,
            lora_rank=self.config.lora_rank if self.config.use_lora else 0,
        )
        self.up_proj = LoRALinear(
            features=self.config.intermediate_size,
            dtype=self.dtype,
            lora_rank=self.config.lora_rank if self.config.use_lora else 0,
        )
        self.down_proj = LoRALinear(
            features=self.config.hidden_size,
            dtype=self.dtype,
            lora_rank=self.config.lora_rank if self.config.use_lora else 0,
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class LLaMABlock(nn.Module):
    """LLaMA transformer block."""

    config: LLaMAConfig
    dtype: jnp.dtype = jnp.bfloat16

    def setup(self):
        self.attention = LLaMAAttention(self.config, dtype=self.dtype)
        self.mlp = LLaMAMLP(self.config, dtype=self.dtype)
        self.input_layernorm = RMSNorm(
            self.config.hidden_size, eps=self.config.rms_norm_eps, dtype=self.dtype
        )
        self.post_attention_layernorm = RMSNorm(
            self.config.hidden_size, eps=self.config.rms_norm_eps, dtype=self.dtype
        )

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        # Self attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            deterministic=deterministic,
        )
        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class LLaMAForCausalLM(nn.Module):
    """LLaMA model for causal language modeling."""

    config: LLaMAConfig
    dtype: jnp.dtype = jnp.bfloat16

    def setup(self):
        self.embed_tokens = nn.Embed(
            self.config.vocab_size, self.config.hidden_size, dtype=self.dtype
        )

        self.layers = [
            LLaMABlock(self.config, dtype=self.dtype)
            for _ in range(self.config.num_hidden_layers)
        ]

        self.norm = RMSNorm(
            self.config.hidden_size, eps=self.config.rms_norm_eps, dtype=self.dtype
        )

    def __call__(
        self,
        input_ids: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        hidden_states = self.embed_tokens(input_ids)

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                deterministic=deterministic,
            )

        hidden_states = self.norm(hidden_states)
        logits = self.embed_tokens.attend(hidden_states)

        return logits
