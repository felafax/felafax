from typing import Any, Dict, Optional, Tuple
import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.spmd as xs
import torch_xla.distributed.xla_multiprocessing as xmp
from ....core.base_classes import BaseDistributedTrainer, TrainingConfig
from ..config import LLaMAConfig


class LLaMAPyTorchTrainer(BaseDistributedTrainer):
    """PyTorch XLA-based distributed trainer for LLaMA models."""

    def __init__(
        self,
        config: LLaMAConfig,
        training_config: TrainingConfig,
        dist_config: Any,
    ):
        self.config = config
        self.training_config = training_config
        self.dist_config = dist_config
        self.device = xm.xla_device()

    def setup_distributed(self) -> None:
        """Initialize distributed training environment."""
        # Create mesh for model partitioning
        num_devices = xm.xrt_world_size()
        mesh_shape = (1, num_devices, 1)  # (dp, fsdp, mp)
        device_ids = list(range(num_devices))
        self.mesh = xs.Mesh(device_ids, mesh_shape, ("dp", "fsdp", "mp"))

    def create_optimizer(
        self, model: torch.nn.Module, learning_rate: float
    ) -> torch.optim.Optimizer:
        """Create optimizer with proper parameter groups."""
        # Separate LoRA parameters from base parameters
        lora_params = []
        base_params = []

        for name, param in model.named_parameters():
            if "lora_" in name:
                lora_params.append(param)
            else:
                base_params.append(param)

        # Create parameter groups with different learning rates
        param_groups = [
            {"params": base_params, "lr": learning_rate, "weight_decay": 0.01},
            {
                "params": lora_params,
                "lr": learning_rate * 10,  # Higher learning rate for LoRA
                "weight_decay": 0.0,  # No weight decay for LoRA
            },
        ]

        return torch.optim.AdamW(param_groups)

    def train_step(
        self,
        model: torch.nn.Module,
        batch: Dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer,
    ) -> Dict[str, float]:
        """Execute single training step."""
        model.train()
        optimizer.zero_grad()

        # Mark input sharding
        xs.mark_sharding(batch["input_ids"], self.mesh, (0, 1))
        xs.mark_sharding(batch["attention_mask"], self.mesh, (0, 1))
        if "labels" in batch:
            xs.mark_sharding(batch["labels"], self.mesh, (0, 1))

        # Forward pass
        outputs = model(**batch)
        loss = outputs["loss"]

        # Backward pass
        loss.backward()

        # Optimizer step
        optimizer.step()
        xm.mark_step()

        # Compute metrics
        metrics = {
            "loss": loss.item(),
        }
        if "logits" in outputs:
            predictions = outputs["logits"].argmax(-1)
            correct = (predictions == batch["labels"]).float().mean()
            metrics["accuracy"] = correct.item()

        return metrics

    def train_epoch(
        self,
        model: torch.nn.Module,
        dataloader: Any,
        optimizer: torch.optim.Optimizer,
    ) -> Dict[str, float]:
        """Train for one epoch."""
        epoch_metrics = {
            "loss": 0.0,
            "accuracy": 0.0,
        }
        steps = 0

        # Create parallel loader
        para_loader = pl.ParallelLoader(dataloader, [self.device]).per_device_loader(
            self.device
        )

        # Training loop
        for batch in para_loader:
            step_metrics = self.train_step(model, batch, optimizer)

            # Accumulate metrics
            for k, v in step_metrics.items():
                epoch_metrics[k] += v
            steps += 1

            # Log progress
            if steps % self.training_config.print_every_n_steps == 0:
                log_metrics = {k: v / steps for k, v in epoch_metrics.items()}
                xm.master_print(f"Step {steps}: {log_metrics}")

            # Break if max steps reached
            if (
                self.training_config.max_steps
                and steps >= self.training_config.max_steps
            ):
                break

        # Average metrics
        for k in epoch_metrics:
            epoch_metrics[k] /= steps

        return epoch_metrics

    def evaluate(
        self,
        model: torch.nn.Module,
        dataloader: Any,
    ) -> Dict[str, float]:
        """Evaluate model."""
        model.eval()
        eval_metrics = {
            "loss": 0.0,
            "accuracy": 0.0,
        }
        steps = 0

        # Create parallel loader
        para_loader = pl.ParallelLoader(dataloader, [self.device]).per_device_loader(
            self.device
        )

        # Evaluation loop
        with torch.no_grad():
            for batch in para_loader:
                # Mark input sharding
                xs.mark_sharding(batch["input_ids"], self.mesh, (0, 1))
                xs.mark_sharding(batch["attention_mask"], self.mesh, (0, 1))
                if "labels" in batch:
                    xs.mark_sharding(batch["labels"], self.mesh, (0, 1))

                # Forward pass
                outputs = model(**batch)
                loss = outputs["loss"]

                # Compute metrics
                step_metrics = {
                    "loss": loss.item(),
                }
                if "logits" in outputs:
                    predictions = outputs["logits"].argmax(-1)
                    correct = (predictions == batch["labels"]).float().mean()
                    step_metrics["accuracy"] = correct.item()

                # Accumulate metrics
                for k, v in step_metrics.items():
                    eval_metrics[k] += v
                steps += 1

                # Break if max eval steps reached
                if (
                    self.training_config.max_eval_steps
                    and steps >= self.training_config.max_eval_steps
                ):
                    break

        # Average metrics
        for k in eval_metrics:
            eval_metrics[k] /= steps

        return eval_metrics

    def generate(
        self,
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        """Generate text using the model."""
        model.eval()

        # Move input to device
        input_ids = input_ids.to(self.device)

        # Initialize sequence with input_ids
        sequence = input_ids

        with torch.no_grad():
            for _ in range(max_length):
                # Get logits for next token
                outputs = model(input_ids=sequence)
                next_token_logits = outputs["logits"][:, -1, :]

                # Apply temperature
                if temperature > 0:
                    next_token_logits = next_token_logits / temperature

                # Apply top-k
                if top_k > 0:
                    indices_to_remove = (
                        next_token_logits
                        < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    )
                    next_token_logits[indices_to_remove] = float("-inf")

                # Apply top-p (nucleus sampling)
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(
                        next_token_logits, descending=True
                    )
                    cumulative_probs = torch.cumsum(
                        torch.softmax(sorted_logits, dim=-1), dim=-1
                    )
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                        ..., :-1
                    ].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        dim=-1, index=sorted_indices, src=sorted_indices_to_remove
                    )
                    next_token_logits[indices_to_remove] = float("-inf")

                # Sample next token
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # Append to sequence
                sequence = torch.cat([sequence, next_token], dim=-1)

                # Break if EOS token generated
                if next_token.item() == self.config.eos_token_id:
                    break

                # Mark step for XLA
                xm.mark_step()

        return sequence
