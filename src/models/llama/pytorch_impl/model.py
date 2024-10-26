import math
from typing import Optional, Tuple, Dict, Any
import torch
import torch.nn as nn
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
from ..config import LLaMAConfig
from ....core.distributed import DistributedConfig


class RotaryEmbedding(nn.Module):
    """Rotary position embeddings."""

    def __init__(
        self, dim: int, max_position_embeddings: int = 2048, base: int = 10000
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # Create inverse frequency bands
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Create position embeddings cache
        self._set_cos_sin_cache(max_position_embeddings, torch.float32)

    def _set_cos_sin_cache(self, seq_len: int, dtype: torch.dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer(
            "cos_cached", emb.cos()[None, None, :, :], persistent=False
        )
        self.register_buffer(
            "sin_cached", emb.sin()[None, None, :, :], persistent=False
        )

    def forward(self, x: torch.Tensor, seq_len: int):
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len, x.dtype)

        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


class LoRALinear(nn.Module):
    """Linear layer with LoRA."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.lora_dropout = nn.Dropout(p=dropout)
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        self.rank = rank
        self.alpha = alpha

        # Initialize LoRA weights
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_output = self.linear(x)
        if self.rank > 0:
            lora_output = (self.lora_dropout(x) @ self.lora_A @ self.lora_B) * (
                self.alpha / self.rank
            )
            return base_output + lora_output
        return base_output


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class LLaMAAttention(nn.Module):
    """Multi-head attention with support for GQA."""

    def __init__(self, config: LLaMAConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads

        # Projection layers with optional LoRA
        if config.use_lora:
            linear_cls = lambda *args, **kwargs: LoRALinear(
                *args,
                **kwargs,
                rank=config.lora_rank,
                alpha=config.lora_alpha,
                dropout=config.lora_dropout
            )
        else:
            linear_cls = nn.Linear

        self.q_proj = linear_cls(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=False
        )
        self.k_proj = linear_cls(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = linear_cls(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = linear_cls(config.hidden_size, config.hidden_size, bias=False)

        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        batch_size, seq_length = hidden_states.shape[:2]

        # Project to query, key, value
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape and apply rotary embeddings
        query_states = query_states.view(
            batch_size, seq_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            batch_size, seq_length, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            batch_size, seq_length, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        # Apply rotary embeddings
        cos, sin = self.rotary_emb(value_states, seq_length)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )

        # Repeat key/value for multi-query attention if needed
        if self.num_key_value_heads != self.num_heads:
            key_states = repeat_kv(
                key_states, self.num_heads // self.num_key_value_heads
            )
            value_states = repeat_kv(
                value_states, self.num_heads // self.num_key_value_heads
            )

        # Compute attention
        attn_weights = torch.matmul(
            query_states, key_states.transpose(-2, -1)
        ) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = torch.softmax(attn_weights, dim=-1)

        # Compute attention output
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_length, self.hidden_size)

        attn_output = self.o_proj(attn_output)
        return attn_output


class LLaMAMLP(nn.Module):
    """LLaMA MLP module."""

    def __init__(self, config: LLaMAConfig):
        super().__init__()
        if config.use_lora:
            linear_cls = lambda *args, **kwargs: LoRALinear(
                *args,
                **kwargs,
                rank=config.lora_rank,
                alpha=config.lora_alpha,
                dropout=config.lora_dropout
            )
        else:
            linear_cls = nn.Linear

        self.gate_proj = linear_cls(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = linear_cls(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = linear_cls(
            config.intermediate_size, config.hidden_size, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class LLaMABlock(nn.Module):
    """LLaMA transformer block."""

    def __init__(self, config: LLaMAConfig):
        super().__init__()
        self.attention = LLaMAAttention(config)
        self.mlp = LLaMAMLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class LLaMAForCausalLM(nn.Module):
    """LLaMA model with causal language modeling head."""

    def __init__(self, config: LLaMAConfig):
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [LLaMABlock(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        if config.tie_word_embeddings:
            self.lm_head = lambda x: torch.matmul(
                x, self.embed_tokens.weight.transpose(0, 1)
            )
        else:
            if config.use_lora:
                self.lm_head = LoRALinear(
                    config.hidden_size,
                    config.vocab_size,
                    rank=config.lora_rank,
                    alpha=config.lora_alpha,
                    dropout=config.lora_dropout,
                    bias=False,
                )
            else:
                self.lm_head = nn.Linear(
                    config.hidden_size, config.vocab_size, bias=False
                )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(
                module.weight, mean=0.0, std=self.config.initializer_range
            )
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(
                module.weight, mean=0.0, std=self.config.initializer_range
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        # Handle position IDs
        if position_ids is None:
            position_ids = torch.arange(
                0, input_ids.shape[1], dtype=torch.long, device=input_ids.device
            )
            position_ids = position_ids.unsqueeze(0)

        # Encode inputs
        hidden_states = self.embed_tokens(input_ids)

        # Apply transformer layers
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )

        # Apply final norm and compute logits
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

        return {
            "loss": loss,
            "logits": logits,
            "hidden_states": hidden_states,
        }
