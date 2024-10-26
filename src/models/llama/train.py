import os
from typing import Optional, Dict, Any
import json
from pathlib import Path
import torch
import jax
from ...core.base_classes import TrainingConfig
from ...core.distributed import BackendType, DistributedConfig
from .config import LLaMAConfig
from .model_handler import LLaMAModelHandler
from .dataset_handler import LLaMADatasetHandler
from .training_utils import (
    setup_wandb,
    log_metrics,
    save_training_state,
    load_training_state,
    get_parameter_count,
)


def train(
    model_name: str,
    data_path: str,
    output_dir: str,
    backend: BackendType,
    training_config: Optional[TrainingConfig] = None,
    model_config: Optional[Dict[str, Any]] = None,
    hf_token: Optional[str] = None,
    use_wandb: bool = True,
    resume_from: Optional[str] = None,
) -> None:
    """High-level training pipeline for LLaMA models."""

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize configs
    training_config = training_config or TrainingConfig()
    llama_config = LLaMAConfig(**(model_config or {}))

    # Initialize distributed config
    if backend == BackendType.JAX:
        num_devices = len(jax.devices())
    else:
        import torch_xla.core.xla_model as xm

        num_devices = xm.xrt_world_size()

    dist_config = DistributedConfig(backend=backend, num_devices=num_devices)

    # Initialize wandb if requested
    if use_wandb:
        setup_wandb(training_config, model_name)

    # Initialize model handler
    model_handler = LLaMAModelHandler(
        model_name=model_name,
        backend=backend,
        hf_token=hf_token,
        base_dir=str(output_dir / "model_cache"),
    )

    # Load model and tokenizer
    model, tokenizer = model_handler.load_model()

    # Initialize dataset handler
    dataset_handler = LLaMADatasetHandler(backend=backend)

    # Get datasets
    train_dataloader, eval_dataloader = dataset_handler.get_dataset(
        data_path, tokenizer, training_config
    )

    # Initialize appropriate trainer
    if backend == BackendType.JAX:
        from .jax_impl.trainer import LLaMAJAXTrainer as Trainer
    else:
        from .pytorch_impl.trainer import LLaMAPyTorchTrainer as Trainer

    trainer = Trainer(
        config=llama_config,
        training_config=training_config,
        dist_config=dist_config,
    )

    # Setup distributed training
    trainer.setup_distributed()

    # Load checkpoint if resuming
    start_epoch = 0
    if resume_from:
        state, loaded_config = load_training_state(resume_from, backend)
        model_handler.load_checkpoint(state)
        training_config = loaded_config
        start_epoch = training_config.num_epochs

    # Print model statistics
    param_counts = get_parameter_count(model, backend)
    print(f"Model Statistics:")
    print(f"Total Parameters: {param_counts['total_parameters']:,}")
    print(f"Trainable Parameters: {param_counts['trainable_parameters']:,}")

    # Training loop
    for epoch in range(start_epoch, training_config.num_epochs):
        print(f"\nEpoch {epoch + 1}/{training_config.num_epochs}")

        # Train
        train_metrics = trainer.train_epoch(
            model,
            train_dataloader,
            optimizer if backend == BackendType.PYTORCH_XLA else None,
        )
        log_metrics(train_metrics, epoch, prefix="train", use_wandb=use_wandb)

        # Evaluate
        eval_metrics = trainer.evaluate(model, eval_dataloader)
        log_metrics(eval_metrics, epoch, prefix="eval", use_wandb=use_wandb)

        # Save checkpoint
        checkpoint_path = save_training_state(
            output_dir / "checkpoints", model, backend, epoch, training_config
        )
        print(f"Saved checkpoint to {checkpoint_path}")

    print("\nTraining complete!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train LLaMA models")
    parser.add_argument(
        "--model_name", required=True, help="Name of the model to train"
    )
    parser.add_argument("--data_path", required=True, help="Path to training data")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--backend", choices=["jax", "pytorch_xla"], default="jax")
    parser.add_argument("--config", help="Path to training configuration file")
    parser.add_argument("--hf_token", help="HuggingFace token")
    parser.add_argument("--no_wandb", action="store_true", help="Disable W&B logging")
    parser.add_argument("--resume_from", help="Path to checkpoint to resume from")

    args = parser.parse_args()

    # Load training config if provided
    training_config = None
    if args.config:
        with open(args.config) as f:
            config_dict = json.load(f)
            training_config = TrainingConfig(**config_dict)

    # Set backend type
    backend = BackendType.JAX if args.backend == "jax" else BackendType.PYTORCH_XLA

    train(
        model_name=args.model_name,
        data_path=args.data_path,
        output_dir=args.output_dir,
        backend=backend,
        training_config=training_config,
        hf_token=args.hf_token,
        use_wandb=not args.no_wandb,
        resume_from=args.resume_from,
    )
