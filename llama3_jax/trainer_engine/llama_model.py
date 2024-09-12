from typing import Any, Dict, List, Optional, Tuple, Union

import einops
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.linen import combine_masks, make_causal_mask
from flax.linen.attention import dot_product_attention_weights
from jax.sharding import PartitionSpec as PS
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_flax_outputs import (FlaxBaseModelOutput,
                                                FlaxCausalLMOutput)

from . import jax_utils


class RMSNorm(nn.Module):
    dim: int
    eps: float = 1e-6
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    def setup(self) -> None:
        self.weight = self.param(
            "kernel",
            nn.initializers.ones,
            (self.dim, ),
            self.param_dtype,
        )

    def _norm(self, x: jnp.ndarray) -> jnp.ndarray:
        return x * jax.lax.rsqrt(
            jnp.square(x).mean(-1, keepdims=True) + self.eps)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = x.astype(jnp.promote_types(self.dtype, jnp.float32))
        output = self._norm(x).astype(self.dtype)
        weight = jnp.asarray(self.weight, self.dtype)
        return output * weight


def apply_rotary_emb(
    xq: jnp.ndarray,
    xk: jnp.ndarray,
    position_ids: jnp.ndarray,
    max_pos: int,
    theta: float = 10000.0,
):
    input_dtype = xq.dtype

    with jax.ensure_compile_time_eval():
        dim = xq.shape[-1]
        freqs = 1.0 / (theta**(
            jnp.arange(0, dim, 2)[:(dim // 2)].astype(jnp.float32) / dim))
        t = jnp.arange(max_pos)
        freqs = jnp.outer(t, freqs).astype(jnp.float32)
        sin, cos = jnp.sin(freqs), jnp.cos(freqs)
        freqs_cis = jnp.complex64(cos + 1j * sin)

    freqs_cis = jnp.take(freqs_cis, position_ids, axis=0)
    reshape_xq = xq.astype(jnp.float32).reshape(*xq.shape[:-1], -1, 2)
    reshape_xk = xk.astype(jnp.float32).reshape(*xk.shape[:-1], -1, 2)

    xq_ = jax.lax.complex(reshape_xq[..., 0], reshape_xq[..., 1])
    xk_ = jax.lax.complex(reshape_xk[..., 0], reshape_xk[..., 1])
    # add head dim
    freqs_cis = jnp.reshape(freqs_cis,
                            (*freqs_cis.shape[:2], 1, *freqs_cis.shape[2:]))
    xq_out = xq_ * freqs_cis
    xq_out = jnp.stack((jnp.real(xq_out), jnp.imag(xq_out)),
                       axis=-1).reshape(*xq_out.shape[:-1], -1)
    xk_out = xk_ * freqs_cis
    xk_out = jnp.stack((jnp.real(xk_out), jnp.imag(xk_out)),
                       axis=-1).reshape(*xk_out.shape[:-1], -1)
    return xq_out.astype(input_dtype), xk_out.astype(input_dtype)


class Attention(nn.Module):
    """
    Attention module for the LLaMA model.
    Implements multi-head attention with support for grouped-query attention.
    """

    config: PretrainedConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self):
        config = self.config
        head_dim = config.hidden_size // config.num_attention_heads

        # Define the query, key, value, and output projection layers
        # Note: In grouped-query attention, we have fewer key-value heads than query heads
        self.wq = nn.Dense(
            config.num_attention_heads * head_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(
                self.config.initializer_range / np.sqrt(config.hidden_size)),
            precision=self.precision,
        )
        self.wk = nn.Dense(
            config.num_key_value_heads * head_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(
                self.config.initializer_range / np.sqrt(config.hidden_size)),
            precision=self.precision,
        )
        self.wv = nn.Dense(
            config.num_key_value_heads * head_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(
                self.config.initializer_range / np.sqrt(config.hidden_size)),
            precision=self.precision,
        )
        self.wo = nn.Dense(
            config.hidden_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(
                self.config.initializer_range / np.sqrt(config.hidden_size)),
            precision=self.precision,
        )

        # Dropout layer for regularization
        self.resid_dropout = nn.Dropout(rate=config.residue_dropout)

    def __call__(
        self,
        hidden_states,
        attention_mask,
        position_ids,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
        fcm_mask=None,
    ):
        # Project input hidden states to query, key, and value
        xq, xk, xv = (
            self.wq(hidden_states),
            self.wk(hidden_states),
            self.wv(hidden_states),
        )

        # Apply sharding constraints for model partitioning.
        xq = jax_utils.apply_sharding_constraint(
            xq, PS(("dp", "fsdp"), None, "mp"))
        xk = jax_utils.apply_sharding_constraint(
            xk, PS(("dp", "fsdp"), None, "mp"))
        xv = jax_utils.apply_sharding_constraint(
            xv, PS(("dp", "fsdp"), None, "mp"))

        # Reshape query for multi-head attention
        xq = einops.rearrange(
            xq,
            "b s (h d) -> b s h d",
            h=self.config.num_attention_heads,
        )
        # Reshape and repeat key and value for grouped-query attention
        xk = einops.repeat(
            xk,
            "b s (h d) -> b s (h g) d",
            h=self.config.num_key_value_heads,
            g=self.config.num_attention_heads //
            self.config.num_key_value_heads,
        )
        xv = einops.repeat(
            xv,
            "b s (h d) -> b s (h g) d",
            h=self.config.num_key_value_heads,
            g=self.config.num_attention_heads //
            self.config.num_key_value_heads,
        )

        # Apply rotary positional embeddings
        xq, xk = apply_rotary_emb(
            xq,
            xk,
            position_ids,
            max_pos=self.config.max_position_embeddings,
        )

        # Set up dropout for attention (if not in deterministic mode)
        dropout_rng = None
        if not deterministic and self.config.attention_dropout > 0.0:
            dropout_rng = self.make_rng("dropout")

        # Create causal mask to ensure the model can only attend to previous tokens
        query_length, key_length = xq.shape[1], xk.shape[1]
        with jax.ensure_compile_time_eval():
            full_causal_mask = make_causal_mask(
                jnp.ones((1, self.config.max_position_embeddings),
                         dtype="bool"),
                dtype="bool",
            )

        causal_mask = full_causal_mask[:, :, :query_length, :key_length]

        # Broadcast causal mask to batch size
        batch_size = hidden_states.shape[0]
        causal_mask = jnp.broadcast_to(causal_mask,
                                       (batch_size, ) + causal_mask.shape[1:])

        # Combine attention mask with causal mask and optional FCM mask
        attention_mask = jnp.broadcast_to(
            jnp.expand_dims(attention_mask, axis=(-3, -2)), causal_mask.shape)
        attention_mask = combine_masks(attention_mask, causal_mask, fcm_mask)

        # Convert boolean mask to float mask for attention computation
        attention_bias = jax.lax.select(
            attention_mask > 0,
            jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
            jnp.full(attention_mask.shape,
                     jnp.finfo(self.dtype).min).astype(self.dtype),
        )

        # Compute attention weights
        attn_weights = dot_product_attention_weights(
            xq,
            xk,
            bias=attention_bias,
            dropout_rng=dropout_rng,
            dropout_rate=self.config.attention_dropout,
            deterministic=deterministic,
            dtype=jnp.promote_types(self.dtype, jnp.float32),
            precision=self.precision,
        )

        # Apply sharding constraint to attention weights
        attn_weights = jax_utils.apply_sharding_constraint(
            attn_weights, PS(("dp", "fsdp"), "mp", None, None))

        # Compute attention output
        attn_output = jnp.einsum("...hqk,...khd->...qhd",
                                 attn_weights,
                                 xv,
                                 precision=self.precision)

        # Reshape attention output and project back to hidden size
        attn_output = einops.rearrange(attn_output, "b s h d -> b s (h d)")
        attn_output = self.wo(attn_output)

        # Apply residual dropout
        attn_output = self.resid_dropout(attn_output,
                                         deterministic=deterministic)

        # Prepare output tuple
        outputs = (attn_output,
                   attn_weights) if output_attentions else (attn_output, )
        return outputs


class FeedForward(nn.Module):
    config: PretrainedConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self) -> None:
        config = self.config
        self.w1 = nn.Dense(
            config.intermediate_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(
                self.config.initializer_range / np.sqrt(config.hidden_size)),
            precision=self.precision,
        )
        self.w2 = nn.Dense(
            config.hidden_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(
                self.config.initializer_range /
                np.sqrt(config.intermediate_size)),
            precision=self.precision,
        )
        self.w3 = nn.Dense(
            config.intermediate_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(
                self.config.initializer_range / np.sqrt(config.hidden_size)),
            precision=self.precision,
        )
        self.dropout = nn.Dropout(rate=self.config.residue_dropout)

    def __call__(self,
                 x: jnp.ndarray,
                 deterministic: bool = True) -> jnp.ndarray:
        x = self.w2(nn.silu(self.w1(x)) * self.w3(x))
        x = self.dropout(x, deterministic=deterministic)
        return x


class TransformerBlock(nn.Module):
    config: PretrainedConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self) -> None:
        self.attention = Attention(
            self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
        )
        self.feed_forward = FeedForward(
            self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
        )
        self.attention_norm = RMSNorm(
            self.config.hidden_size,
            eps=self.config.rms_norm_eps,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        self.ffn_norm = RMSNorm(
            self.config.hidden_size,
            eps=self.config.rms_norm_eps,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )

    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
        fcm_mask: Optional[jnp.ndarray] = None,
    ):
        attn_outputs = self.attention(
            self.attention_norm(hidden_states),
            attention_mask,
            position_ids,
            deterministic,
            init_cache,
            output_attentions,
            fcm_mask,
        )
        attn_output = attn_outputs[0]
        hidden_states = hidden_states + attn_output

        feed_forward_input = self.ffn_norm(hidden_states)

        feed_forward_hidden_states = self.feed_forward(
            feed_forward_input,
            deterministic,
        )
        feed_forward_hidden_states = jax_utils.apply_sharding_constraint(
            feed_forward_hidden_states, PS(("dp", "fsdp"), None, "mp"))

        hidden_states = hidden_states + feed_forward_hidden_states

        return (hidden_states, ) + attn_outputs[1:]


class TransformerBlockCollection(nn.Module):
    config: PretrainedConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self):
        block = TransformerBlock
        self.blocks = [
            block(
                self.config,
                name=str(i),
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision,
            ) for i in range(self.config.num_hidden_layers)
        ]

    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ):
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        fcm_mask = None

        for block in self.blocks:
            if output_hidden_states:
                all_hidden_states += (hidden_states, )

            layer_outputs = block(
                hidden_states,
                attention_mask,
                position_ids,
                deterministic,
                init_cache,
                output_attentions,
                fcm_mask,
            )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions += (layer_outputs[1], )

        # this contains possible `None` values - `FlaxGPTJModule` will filter them out
        outputs = (hidden_states, all_hidden_states, all_attentions)

        return outputs


class LlamaModule(nn.Module):
    config: PretrainedConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self):
        self.embed_dim = self.config.hidden_size

        self.wte = nn.Embed(
            self.config.vocab_size,
            self.config.hidden_size,
            embedding_init=jax.nn.initializers.normal(
                stddev=self.config.initializer_range),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        self.dropout = nn.Dropout(rate=self.config.embedding_dropout)
        self.h = TransformerBlockCollection(
            self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
        )
        self.ln_f = RMSNorm(
            self.config.hidden_size,
            eps=self.config.rms_norm_eps,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )

    def __call__(
        self,
        input_ids,
        attention_mask,
        position_ids,
        deterministic=True,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        input_embeds = self.wte(input_ids.astype("i4"))

        hidden_states = self.dropout(input_embeds, deterministic=deterministic)

        outputs = self.h(
            hidden_states,
            attention_mask,
            position_ids=position_ids,
            deterministic=deterministic,
            init_cache=init_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        hidden_states = outputs[0]
        hidden_states = self.ln_f(hidden_states)

        if output_hidden_states:
            all_hidden_states = outputs[1] + (hidden_states, )
            outputs = (hidden_states, all_hidden_states) + outputs[2:]
        else:
            outputs = (hidden_states, ) + outputs[1:]

        if not return_dict:
            return tuple(v for v in outputs if v is not None)

        return FlaxBaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=outputs[1],
            attentions=outputs[-1],
        )


class CausalLlamaModule(nn.Module):
    config: PretrainedConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self):
        self.transformer = LlamaModule(self.config, dtype=self.dtype)
        self.lm_head = nn.Dense(
            self.config.vocab_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(
                stddev=self.config.initializer_range /
                np.sqrt(self.config.hidden_size)),
            precision=self.precision,
        )

    def __call__(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        batch_size, seq_length = input_ids.shape
        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)
        if position_ids is None:
            position_ids = jnp.broadcast_to(
                jnp.clip(jnp.cumsum(attention_mask, axis=-1) - 1, a_min=0),
                (batch_size, seq_length),
            )
        outputs = self.transformer(
            input_ids,
            attention_mask,
            position_ids,
            deterministic=deterministic,
            init_cache=init_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        lm_logits = self.lm_head(hidden_states)

        if not return_dict:
            return (lm_logits, ) + outputs[1:]

        return FlaxCausalLMOutput(
            logits=lm_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
