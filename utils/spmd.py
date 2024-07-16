import torch
import torch.nn as nn
import re
import torch_xla.experimental.xla_sharding as xs
import torch_xla.core.xla_model as xm
from transformers import LlamaConfig

# Sharding rules for LLaMA model
# Each tuple contains (regex_pattern, (shard_dim1, shard_dim2))
# 0: data parallel, 1: fully sharded data parallel, 2: model parallel
LLAMA_RULES = [
    ("model\\.embed_tokens", ("mp", "fsdp")),  # Embedding layer
    ("self_attn\\.(q_proj|k_proj|v_proj)", ("fsdp", "mp")),  # Self-attention projections
    ("self_attn\\.o_proj", ("mp", "fsdp")),  # Self-attention output projection
    ("mlp\\.gate_proj", ("fsdp", "mp")),  # MLP gate projection
    ("mlp\\.down_proj", ("mp", "fsdp")),  # MLP down projection
    ("mlp\\.up_proj", ("fsdp", "mp")),  # MLP up projection
    ("lm_head", ("fsdp", "mp")),  # Language model head
]

def partition_module(model, mesh, device=xm.xla_device(), verbose=False):
    """
    Partition a LLaMA model according to the defined sharding rules.
    
    Args:
    - model: The LLaMA model to be partitioned
    - mesh: The device mesh for sharding
    - device: The XLA device (default: xm.xla_device())
    - verbose: If True, print matching rules (default: False)
    """
    
    if not isinstance(model.config, LlamaConfig):
        raise Exception("This function only supports LLaMA models")

    for name, module in model.named_modules():
        module.to(device)
        
        if isinstance(module, (nn.Embedding, nn.Linear)):
            for rule_pattern, spec in LLAMA_RULES:
                if re.findall(rule_pattern, name):
                    if verbose:
                        print(f"Matching rule {rule_pattern} for module {name}")
                    
                    # Convert string spec to numerical spec
                    numerical_spec = tuple(["dp", "fsdp", "mp"].index(s) for s in spec)
                    xs.mark_sharding(module.weight, mesh, numerical_spec)
                    break

def partition_module_dp(model, mesh, device=xm.xla_device(), verbose=False):
    """
    Partition a model using simple data parallelism.
    
    Args:
    - model: The model to be partitioned
    - mesh: The device mesh for sharding
    - device: The XLA device (default: xm.xla_device())
    - verbose: If True, print partitioned modules (default: False)
    """
   
    spec = ("dp", "mp")  # Simple data parallel spec

    for name, module in model.named_modules():
        module.to(device)
        if isinstance(module, (nn.Embedding, nn.Linear)):
            numerical_spec = tuple(["dp", "fsdp", "mp"].index(s) for s in spec)
            xs.mark_sharding(module.weight, mesh, numerical_spec)
            if verbose:
                print(f"Partitioning module {name} with data parallelism")