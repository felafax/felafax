#!/usr/bin/env python3
import os
import sys
import argparse
import json
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any


def setup_environment(base_dir: str) -> None:
    """Setup environment variables."""
    os.environ["HF_HUB_CACHE"] = os.path.join(base_dir, "hf_cache")
    os.environ["HF_HOME"] = os.path.join(base_dir, "hf_home")
    os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/local/cuda"

    if "TPU_NAME" in os.environ:
        os.environ["PJRT_DEVICE"] = "TPU"

    # JAX-specific settings
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
        os.environ["JAX_PLATFORM_NAME"] = "gpu"

    # Create cache directories
    Path(os.environ["HF_HUB_CACHE"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["HF_HOME"]).mkdir(parents=True, exist_ok=True)


def get_world_size() -> int:
    """Get number of devices for distributed training."""
    if "TPU_NAME" in os.environ:
        import torch_xla.core.xla_model as xm

        return xm.xrt_world_size()
    elif "CUDA_VISIBLE_DEVICES" in os.environ:
        return len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
    else:
        import jax

        return len(jax.devices())


def launch_jax(
    args: argparse.Namespace,
    world_size: int,
) -> None:
    """Launch JAX training process."""
    cmd = [
        sys.executable,
        "-m",
        "src.models.llama.train",
        "--model_name",
        args.model_name,
        "--data_path",
        args.data_path,
        "--output_dir",
        args.output_dir,
        "--backend",
        "jax",
    ]

    if args.config:
        cmd.extend(["--config", args.config])
    if args.hf_token:
        cmd.extend(["--hf_token", args.hf_token])
    if args.no_wandb:
        cmd.append("--no_wandb")
    if args.resume_from:
        cmd.extend(["--resume_from", args.resume_from])

    env = os.environ.copy()
    subprocess.run(cmd, env=env, check=True)


def launch_pytorch_xla(
    args: argparse.Namespace,
    world_size: int,
) -> None:
    """Launch PyTorch XLA training process."""
    cmd = [
        sys.executable,
        "-m",
        "torch_xla.distributed.xla_dist",
        "--tpu",
        os.environ.get("TPU_NAME", ""),
        "--conda-env",
        os.environ.get("CONDA_DEFAULT_ENV", ""),
        "--env",
        "HF_HOME=" + os.environ["HF_HOME"],
        "--env",
        "HF_HUB_CACHE=" + os.environ["HF_HUB_CACHE"],
        "--",
        "src.models.llama.train",
        "--model_name",
        args.model_name,
        "--data_path",
        args.data_path,
        "--output_dir",
        args.output_dir,
        "--backend",
        "pytorch_xla",
    ]

    if args.config:
        cmd.extend(["--config", args.config])
    if args.hf_token:
        cmd.extend(["--hf_token", args.hf_token])
    if args.no_wandb:
        cmd.append("--no_wandb")
    if args.resume_from:
        cmd.extend(["--resume_from", args.resume_from])

    env = os.environ.copy()
    subprocess.run(cmd, env=env, check=True)


def main():
    parser = argparse.ArgumentParser(description="Launch distributed training")
    parser.add_argument(
        "--model_name", required=True, help="Name of the model to train"
    )
    parser.add_argument("--data_path", required=True, help="Path to training data")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument(
        "--base_dir", default="/tmp/llama", help="Base directory for cache"
    )
    parser.add_argument("--backend", choices=["jax", "pytorch_xla"], required=True)
    parser.add_argument("--config", help="Path to training configuration file")
    parser.add_argument("--hf_token", help="HuggingFace token")
    parser.add_argument("--no_wandb", action="store_true", help="Disable W&B logging")
    parser.add_argument("--resume_from", help="Path to checkpoint to resume from")

    args = parser.parse_args()

    # Setup environment
    setup_environment(args.base_dir)

    # Get world size for distributed training
    world_size = get_world_size()
    print(f"Launching training with {world_size} devices")

    # Launch appropriate backend
    if args.backend == "jax":
        launch_jax(args, world_size)
    else:
        launch_pytorch_xla(args, world_size)


if __name__ == "__main__":
    main()
