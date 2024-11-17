"""Equinox implementation of the Llama model with LoRA support."""

import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Optional


class LlamaEmbedding(eqx.Module):
    weight: jnp.ndarray

    def __init__(self, num_embeddings, embedding_dim):
        self.weight = jax.random.normal(
            jax.random.PRNGKey(0), (num_embeddings, embedding_dim)
        )

    def __call__(self, x):
        return jnp.take(self.weight, x, axis=0)


# TODO(lora): Need to change this to equnix.linear or define my own quax.
# TODO(lora): Remove static=True to do full fine-tuning.
class LlamaLinear(eqx.Module):
    weight: jnp.ndarray
    bias: Optional[jnp.ndarray]
    lora_A: Optional[jnp.ndarray]
    lora_B: Optional[jnp.ndarray]

    def __init__(self, in_features, out_features, bias=False, rank=0, key=None):
        if key is not None:
            keys = jax.random.split(key, 4)
        else:
            keys = jax.random.split(jax.random.PRNGKey(99), 4)
        self.weight = jax.random.normal(
            keys[0],
            (out_features, in_features),
        )
        self.bias = jax.random.normal(keys[1], (out_features,)) if bias else None
        if rank > 0:
            self.lora_A = jax.random.normal(keys[2], (in_features, rank))
            self.lora_B = jax.random.normal(keys[3], (rank, out_features))
        else:
            self.lora_A = None
            self.lora_B = None

    def __call__(self, x):
        y = x @ self.weight.T
        if self.bias is not None:
            y += self.bias
        if self.lora_A is not None and self.lora_B is not None:
            lora_update = x @ self.lora_A @ self.lora_B
            y += lora_update
        return y


class LlamaRotaryEmbedding(eqx.Module):
    inv_freq: jnp.ndarray
    max_seq_len_cached: int

    def __init__(self, config):
        dim = config.hidden_size // config.num_attention_heads
        self.max_seq_len_cached = config.max_position_embeddings
        inv_freq = 1.0 / (
            config.rope_theta
            ** (jnp.arange(0, dim, 2).astype(jnp.float32) / dim)
        )
        self.inv_freq = inv_freq

    def __call__(self, x, position_ids):
        seq_len = position_ids.shape[1]
        t = position_ids.astype(jnp.float32)
        inv_freq = self.inv_freq

        # Reshape t to match the expected input shape
        t = t.reshape(-1, seq_len, 1)  # Shape: (batch_size, seq_len, 1)

        # Compute freqs directly without using einsum
        freqs = (
            t * inv_freq[None, None, :]
        )  # Shape: (batch_size, seq_len, dim//2)

        emb = jnp.concatenate(
            (freqs, freqs), axis=-1
        )  # Shape: (batch_size, seq_len, dim)
        cos = jnp.cos(emb)
        sin = jnp.sin(emb)
        return cos.astype(x.dtype), sin.astype(x.dtype)


class LlamaRMSNorm(eqx.Module):
    weight: jnp.ndarray
    eps: float

    def __init__(self, hidden_size, eps=1e-6):
        self.weight = jnp.ones(hidden_size)
        self.eps = eps

    def __call__(self, hidden_states):
        variance = jnp.mean(hidden_states**2, axis=-1, keepdims=True)
        hidden_states = hidden_states * jax.lax.rsqrt(variance + self.eps)
        return self.weight * hidden_states


class LlamaSdpaAttention(eqx.Module):
    q_proj: LlamaLinear
    k_proj: LlamaLinear
    v_proj: LlamaLinear
    o_proj: LlamaLinear
    rotary_emb: LlamaRotaryEmbedding
    num_heads: int
    num_key_value_heads: int
    num_key_value_groups: int
    head_dim: int
    hidden_size: int
    max_position_embeddings: int
    rope_theta: float

    def __init__(self, config, key):
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        keys = jax.random.split(key, 4)
        self.q_proj = LlamaLinear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            bias=config.attention_bias,
            rank=config.lora_rank,
            key=keys[0],
        )
        self.k_proj = LlamaLinear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
            rank=config.lora_rank,
            key=keys[1],
        )
        self.v_proj = LlamaLinear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
            rank=config.lora_rank,
            key=keys[2],
        )
        self.o_proj = LlamaLinear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=config.attention_bias,
            rank=config.lora_rank,
            key=keys[3],
        )

        self.rotary_emb = LlamaRotaryEmbedding(config)

    def __call__(self, x, position_ids, attention_mask=None):
        def jax_apply_rotary_pos_emb(q, k, cos, sin):
            q_embed = (q * cos) + (jax_rotate_half(q) * sin)
            k_embed = (k * cos) + (jax_rotate_half(k) * sin)
            return q_embed, k_embed

        def jax_rotate_half(x):
            x1, x2 = jnp.split(x, 2, axis=-1)
            return jnp.concatenate((-x2, x1), axis=-1)

        bsz, q_len, _ = x.shape

        query_states = self.q_proj(x)
        key_states = self.k_proj(x)
        value_states = self.v_proj(x)

        query_states = query_states.reshape(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(0, 2, 1, 3)
        key_states = key_states.reshape(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(0, 2, 1, 3)
        value_states = value_states.reshape(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(0, 2, 1, 3)

        cos, sin = self.rotary_emb(value_states, position_ids)

        # Reshape cos and sin to include num_heads dimension
        cos = cos[:, None, :, :]  # (batch_size, 1, seq_len, head_dim)
        sin = sin[:, None, :, :]  # (batch_size, 1, seq_len, head_dim)

        # Apply rotary embeddings
        query_states, key_states = jax_apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        if self.num_key_value_heads != self.num_heads:
            key_states = jnp.repeat(
                key_states, self.num_heads // self.num_key_value_heads, axis=1
            )
            value_states = jnp.repeat(
                value_states, self.num_heads // self.num_key_value_heads, axis=1
            )

        attn_weights = jnp.einsum(
            "bhqd,bhkd->bhqk", query_states, key_states
        ) / jnp.sqrt(self.head_dim)

        # Create causal mask
        q_len, k_len = query_states.shape[2], key_states.shape[2]
        causal_mask = jnp.tril(jnp.ones((q_len, q_len)))
        causal_mask = causal_mask[None, None, :, :]

        # If attention_mask is provided, use it to override the causal mask
        if attention_mask is not None:
            # Broadcast attention_mask to the correct shape
            attention_mask = jnp.expand_dims(attention_mask, axis=(1, 2))
            # Combine causal_mask and attention_mask
            combined_mask = jnp.minimum(causal_mask, attention_mask)
        else:
            combined_mask = causal_mask

        # Apply the combined mask
        attn_weights = jnp.where(combined_mask == 0, float("-inf"), attn_weights)

        attn_weights = jax.nn.softmax(attn_weights, axis=-1)

        attn_output = jnp.einsum("bhqk,bhkd->bhqd", attn_weights, value_states)
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(
            bsz, q_len, self.num_heads * self.head_dim
        )

        attn_output = self.o_proj(attn_output)

        return attn_output


class LlamaMLP(eqx.Module):
    gate_proj: LlamaLinear
    up_proj: LlamaLinear
    down_proj: LlamaLinear

    def __init__(self, hidden_size, intermediate_size, key, rank=0):
        keys = jax.random.split(key, 3)
        self.gate_proj = LlamaLinear(
            hidden_size, intermediate_size, bias=False, rank=rank, key=keys[0]
        )
        self.up_proj = LlamaLinear(
            hidden_size, intermediate_size, bias=False, rank=rank, key=keys[1]
        )
        self.down_proj = LlamaLinear(
            intermediate_size, hidden_size, bias=False, rank=rank, key=keys[2]
        )

    def __call__(self, x):
        return self.down_proj(jax.nn.silu(self.gate_proj(x)) * self.up_proj(x))


class LlamaDecoderLayer(eqx.Module):
    self_attn: LlamaSdpaAttention
    mlp: LlamaMLP
    input_layernorm: LlamaRMSNorm
    post_attention_layernorm: LlamaRMSNorm

    def __init__(self, config, key):
        keys = jax.random.split(key, 2)
        self.self_attn = LlamaSdpaAttention(config, keys[0])
        self.mlp = LlamaMLP(
            config.hidden_size,
            config.intermediate_size,
            keys[1],
            config.lora_rank,
        )
        self.input_layernorm = LlamaRMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
        self.post_attention_layernorm = LlamaRMSNorm(
            config.hidden_size, config.rms_norm_eps
        )

    def __call__(self, hidden_states, attention_mask=None, position_ids=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states, position_ids, attention_mask
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class LlamaModel(eqx.Module):
    embed_tokens: LlamaEmbedding
    layers: list[LlamaDecoderLayer]
    norm: LlamaRMSNorm

    def __init__(self, config, key):
        self.embed_tokens = LlamaEmbedding(config.vocab_size, config.hidden_size)
        layer_keys = jax.random.split(key, config.num_hidden_layers)
        self.layers = [
            LlamaDecoderLayer(config, layer_keys[i])
            for i in range(config.num_hidden_layers)
        ]
        self.norm = LlamaRMSNorm(config.hidden_size, config.rms_norm_eps)

    def __call__(self, input_ids, attention_mask=None, position_ids=None):
        hidden_states = self.embed_tokens(input_ids)

        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask, position_ids)

        hidden_states = self.norm(hidden_states)

        return hidden_states


class LlamaForCausalLM(eqx.Module):
    model: LlamaModel
    lm_head: LlamaLinear

    def __init__(self, config, key):
        key1, key2 = jax.random.split(key)
        self.model = LlamaModel(config, key1)
        self.lm_head = LlamaLinear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            rank=config.lora_rank,
            key=key2,
        )

    def __call__(self, input_ids, attention_mask=None, position_ids=None):
        hidden_states = self.model(input_ids, attention_mask, position_ids)
        logits = self.lm_head(hidden_states)
        return logits


class LlamaConfig:
    def __init__(self, **kwargs):
        self.vocab_size = kwargs.get("vocab_size", 32000)
        self.hidden_size = kwargs.get("hidden_size", 4096)
        self.intermediate_size = kwargs.get("intermediate_size", 11008)
        self.num_hidden_layers = kwargs.get("num_hidden_layers", 32)
        self.num_attention_heads = kwargs.get("num_attention_heads", 32)
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 32)
        self.max_position_embeddings = kwargs.get(
            "max_position_embeddings", 2048
        )
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)

        # New attributes
        self.rope_theta = kwargs.get("rope_theta", 10000.0)
        self.attention_bias = kwargs.get("attention_bias", False)
        self.hidden_act = kwargs.get("hidden_act", "silu")
        self.initializer_range = kwargs.get("initializer_range", 0.02)
        self.use_cache = kwargs.get("use_cache", True)
        self.tie_word_embeddings = kwargs.get("tie_word_embeddings", False)
        self.rope_scaling = kwargs.get("rope_scaling", None)

        # Derived attributes
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.pretraining_tp = kwargs.get("pretraining_tp", 1)

        # Optional attributes
        self.bias = kwargs.get(
            "bias", False
        )  # For compatibility with some attention implementations
        self.rope_type = kwargs.get("rope_type", "default")
        self.partial_rotary_factor = kwargs.get("partial_rotary_factor", 1.0)

        # Dropout rates (usually 0.0 for inference)
        self.attention_dropout = kwargs.get("attention_dropout", 0.0)
        self.hidden_dropout = kwargs.get("hidden_dropout", 0.0)

        # Additional optional parameters
        self.bos_token_id = kwargs.get("bos_token_id", None)
        self.eos_token_id = kwargs.get("eos_token_id", None)
        self.pad_token_id = kwargs.get("pad_token_id", None)
        self.torch_dtype = kwargs.get("torch_dtype", None)

        self.lora_rank = kwargs.get("lora_rank", 0)  # Default 0 means no LoRA

    def __repr__(self):
        return f"LlamaConfig({', '.join(f'{k}={v}' for k, v in self.__dict__.items())})"

    def to_dict(self):
        """Serializes the configuration to a dictionary."""
        return self.__dict__
