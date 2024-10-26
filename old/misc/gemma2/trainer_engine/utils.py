import jax
import jax.numpy as jnp
from flax import traverse_util
from flax.core.meta import Partitioned
from typing import Any, Dict

def print_params(params: Dict[str, Any]) -> None:
    flat_params = traverse_util.flatten_dict(params)
    for key, value in flat_params.items():
        name = "/".join(str(x) for x in key)
        print(f"Name: {name}")
        print(f"Shape: {value.shape}")
       
        # jax.debug.visualize_array_sharding doesn't work on higher-order arrays.
        if len(value.shape) <= 2:
            array = value.unbox() if isinstance(value, Partitioned) else value
            print(jax.debug.visualize_array_sharding(array))
        
        print("-" * 40)