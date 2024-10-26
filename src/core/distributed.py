from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Tuple
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as PS
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs


class BackendType(Enum):
    JAX = "jax"
    PYTORCH = "pytorch"
    PYTORCH_XLA = "pytorch_xla"


@dataclass
class ShardingConfig:
    """Configuration for model sharding."""

    mesh_shape: Tuple[int, ...]
    partition_specs: Dict[str, Any]  # Maps parameter types to partition specs


class DistributedConfig:
    """Configuration for distributed training."""

    def __init__(
        self,
        backend: BackendType,
        num_devices: int,
        sharding_config: Optional[ShardingConfig] = None,
    ):
        self.backend = backend
        self.num_devices = num_devices
        self.sharding_config = sharding_config or self._default_sharding_config()

    def _default_sharding_config(self) -> ShardingConfig:
        if self.backend == BackendType.JAX:
            return self._create_jax_sharding()
        elif self.backend == BackendType.PYTORCH_XLA:
            return self._create_pytorch_xla_sharding()
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    def _create_jax_sharding(self) -> ShardingConfig:
        """Create default JAX sharding configuration."""
        if self.num_devices == 1:
            mesh_shape = (1, 1, 1)
        elif self.num_devices == 4:
            mesh_shape = (2, 2, 1)
        elif self.num_devices == 8:
            mesh_shape = (2, 2, 2)
        elif self.num_devices == 16:
            mesh_shape = (1, 4, 4)
        else:
            mesh_shape = (1, self.num_devices, 1)

        partition_specs = {
            "embeddings": PS("mp", "fsdp"),
            "attention": {
                "query": PS("fsdp", "mp"),
                "key": PS("fsdp", "mp"),
                "value": PS("fsdp", "mp"),
                "output": PS("mp", "fsdp"),
            },
            "mlp": {
                "intermediate": PS("fsdp", "mp"),
                "output": PS("mp", "fsdp"),
            },
            "layer_norm": PS(None),
            "lm_head": PS("fsdp", "mp"),
        }

        return ShardingConfig(mesh_shape, partition_specs)

    def _create_pytorch_xla_sharding(self) -> ShardingConfig:
        """Create default PyTorch XLA sharding configuration."""
        mesh_shape = (1, self.num_devices, 1)
        partition_specs = ("dp", "fsdp", "mp")  # PyTorch XLA style
        return ShardingConfig(mesh_shape, partition_specs)

    def create_mesh(self) -> Any:
        """Create appropriate mesh for the backend."""
        if self.backend == BackendType.JAX:
            devices = jax.devices()
            device_mesh = jax.experimental.mesh_utils.create_device_mesh(
                self.sharding_config.mesh_shape
            )
            return Mesh(devices=device_mesh, axis_names=("dp", "fsdp", "mp"))
        elif self.backend == BackendType.PYTORCH_XLA:
            device_ids = list(range(self.num_devices))
            return xs.Mesh(
                device_ids,
                self.sharding_config.mesh_shape,
                self.sharding_config.partition_specs,
            )
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")


def get_device_count(backend: BackendType) -> int:
    """Get number of available devices for given backend."""
    if backend == BackendType.JAX:
        return len(jax.devices())
    elif backend == BackendType.PYTORCH_XLA:
        import torch_xla.runtime as xr

        return xr.global_runtime_device_count()
    elif backend == BackendType.PYTORCH:
        return torch.cuda.device_count()
    else:
        raise ValueError(f"Unsupported backend: {backend}")
