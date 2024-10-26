from typing import Any, Dict, Optional, Tuple
import os
import json
import time
from pathlib import Path
import jax
import jax.numpy as jnp
import torch
import wandb
from tqdm.auto import tqdm
from ...core.base_classes import TrainingConfig
from ...core.distributed import BackendType


def setup_wandb(
    config: TrainingConfig, model_name: str, run_name: Optional[str] = None
) -> None:
    """Initialize Weights & Biases logging."""
    wandb.init(
        project="llama-finetuning",
        name=run_name or f"llama-{model_name}-{time.strftime('%Y%m%d-%H%M%S')}",
        config={"model_name": model_name, **config.__dict__},
    )


def log_metrics(
    metrics: Dict[str, float], step: int, prefix: str = "", use_wandb: bool = True
) -> None:
    """Log metrics to console and W&B."""
    # Add prefix to metric names
    if prefix:
        metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}

    # Print to console
    metrics_str = " ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
    print(f"Step {step}: {metrics_str}")

    # Log to W&B
    if use_wandb:
        wandb.log(metrics, step=step)


def save_training_state(
    save_dir: str,
    state: Any,
    backend: BackendType,
    step: int,
    config: TrainingConfig,
) -> str:
    """Save training state and configuration."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration
    config_path = save_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config.__dict__, f, indent=2)

    # Save checkpoint based on backend
    if backend == BackendType.JAX:
        checkpoint_path = save_dir / f"checkpoint_{step}.msgpack"
        # Only save on process 0
        if jax.process_index() == 0:
            with open(checkpoint_path, "wb") as f:
                import msgpack

                msgpack.dump(state, f)
    elif backend == BackendType.PYTORCH_XLA:
        checkpoint_path = save_dir / f"checkpoint_{step}.pt"
        import torch_xla.core.xla_model as xm

        # Only save on master process
        if xm.is_master_ordinal():
            torch.save(state, checkpoint_path)
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    return str(checkpoint_path)


def load_training_state(
    load_path: str,
    backend: BackendType,
) -> Tuple[Any, TrainingConfig]:
    """Load training state and configuration."""
    load_dir = Path(load_path).parent

    # Load configuration
    config_path = load_dir / "config.json"
    with open(config_path, "r") as f:
        config = TrainingConfig(**json.load(f))

    # Load checkpoint based on backend
    if backend == BackendType.JAX:
        with open(load_path, "rb") as f:
            import msgpack

            state = msgpack.load(f)
    elif backend == BackendType.PYTORCH_XLA:
        state = torch.load(load_path)
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    return state, config


def create_mesh(backend: BackendType, num_devices: Optional[int] = None) -> Any:
    """Create appropriate device mesh based on backend."""
    if backend == BackendType.JAX:
        devices = jax.devices()
        num_devices = num_devices or len(devices)

        # Create mesh shape based on number of devices
        if num_devices == 1:
            mesh_shape = (1, 1)
        elif num_devices == 4:
            mesh_shape = (2, 2)
        elif num_devices == 8:
            mesh_shape = (2, 4)
        else:
            mesh_shape = (1, num_devices)

        device_mesh = jax.experimental.mesh_utils.create_device_mesh(mesh_shape)
        return jax.sharding.Mesh(device_mesh, ("data", "model"))

    elif backend == BackendType.PYTORCH_XLA:
        import torch_xla.core.xla_model as xm
        import torch_xla.distributed.xla_multiprocessing as xmp

        # Get number of devices
        num_devices = num_devices or xm.xrt_world_size()
        return xmp.MpModelWrapper(num_devices)

    else:
        raise ValueError(f"Unsupported backend: {backend}")


def get_parameter_count(params: Any, backend: BackendType) -> Dict[str, int]:
    """Count total and trainable parameters."""
    if backend == BackendType.JAX:

        def count_params(param_dict):
            return sum(x.size for x in jax.tree_leaves(param_dict))

        total_params = count_params(params)
        trainable_params = count_params(
            {k: v for k, v in params.items() if not k.startswith("frozen")}
        )

    elif backend == BackendType.PYTORCH_XLA:

        def count_params(module):
            return sum(p.numel() for p in module.parameters())

        total_params = count_params(params)
        trainable_params = sum(
            p.numel() for p in params.parameters() if p.requires_grad
        )

    else:
        raise ValueError(f"Unsupported backend: {backend}")

    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
    }
