"""Library for partitioning the LLaMa3 model."""

import torch
import torch.nn as nn
import re
import torch_xla.distributed.spmd as xs
import torch_xla.core.xla_model as xm


# Sharding rules for LLaMA model
LLAMA_SHARDING_RULES = [
    ("model\\.embed_tokens", ("mp", "fsdp")),  # Embedding layer
    ("self_attn\\.(q_proj|k_proj|v_proj)", ("fsdp", "mp")),  # Attention layers
    ("self_attn\\.o_proj", ("mp", "fsdp")),  # Attention output projection
    ("mlp\\.gate_proj", ("fsdp", "mp")),  # MLP gate projection
    ("mlp\\.down_proj", ("mp", "fsdp")),  # MLP down projection
    ("mlp\\.up_proj", ("fsdp", "mp")),  # MLP up projection
    ("lm_head", ("fsdp", "mp")),  # Language model head
]


def partition_model(model, mesh, verbose=False):
    """
    Partition a LLaMA model according to the defined sharding rules.

    Args:
    - model: The LLaMA model to be partitioned
    - mesh: The device mesh for sharding
    - verbose: If True, print matching rules (default: False)
    """

    for name, module in model.named_modules():
        if isinstance(module, (nn.Embedding, nn.Linear)):
            for rule_pattern, spec in LLAMA_SHARDING_RULES:
                if re.findall(rule_pattern, name):
                    if verbose:
                        print(f"Matching rule {rule_pattern} for module {name}")

                    # Create partition spec.
                    partition_spec = tuple(["dp", "fsdp", "mp"].index(s) for s in spec)
                    xs.mark_sharding(module.weight, mesh, partition_spec)
                    break
