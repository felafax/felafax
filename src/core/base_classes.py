from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union
import jax
import torch


@dataclass
class TrainingConfig:
    """Base configuration for all trainers."""

    learning_rate: float = 1e-3
    num_epochs: int = 1
    max_steps: Optional[int] = None
    batch_size: int = 16
    seq_length: int = 2048
    dataset_size_limit: Optional[int] = None
    print_every_n_steps: int = 5
    eval_every_n_steps: int = 1000
    max_eval_steps: Optional[int] = 1
    gradient_accumulation_steps: int = 1

    @classmethod
    def from_json(cls, json_path: str) -> "TrainingConfig":
        import json

        with open(json_path, "r") as f:
            config_dict = json.load(f)
        return cls(**config_dict)


@dataclass
class ModelConfig:
    """Base configuration for model architectures."""

    model_name: str
    base_dir: str
    dtype: Any = "bfloat16"
    param_dtype: Any = "bfloat16"


class BaseModelHandler(ABC):
    """Abstract base class for model initialization and handling."""

    @abstractmethod
    def load_model(self) -> Tuple[Any, Any]:
        """Load model and tokenizer."""
        pass

    @abstractmethod
    def save_checkpoint(self, model_state: Any, path: str) -> None:
        """Save model checkpoint."""
        pass

    @abstractmethod
    def load_checkpoint(self, path: str) -> Any:
        """Load model checkpoint."""
        pass


class BaseDistributedTrainer(ABC):
    """Abstract base class for distributed trainers."""

    @abstractmethod
    def setup_distributed(self) -> None:
        """Set up distributed training environment."""
        pass

    @abstractmethod
    def train_step(
        self, state: Any, batch: Dict[str, Any], rng: Optional[Any] = None
    ) -> Tuple[Any, Dict[str, float]]:
        """Execute single training step."""
        pass

    @abstractmethod
    def train_epoch(
        self, state: Any, dataloader: Any, rng: Optional[Any] = None
    ) -> Tuple[Any, Dict[str, float]]:
        """Train for one epoch."""
        pass

    @abstractmethod
    def evaluate(
        self,
        state: Any,
        dataloader: Any,
    ) -> Dict[str, float]:
        """Evaluate model."""
        pass


class BaseDatasetHandler(ABC):
    """Abstract base class for dataset handling."""

    @abstractmethod
    def get_dataset(
        self, data_source: str, tokenizer: Any, config: TrainingConfig
    ) -> Tuple[Any, Any]:
        """Get train and validation dataloaders."""
        pass

    @abstractmethod
    def process_batch(self, batch: Dict[str, Any], tokenizer: Any) -> Dict[str, Any]:
        """Process a single batch of data."""
        pass


# State management for JAX
class JAXTrainState:
    """Training state for JAX models."""

    def __init__(
        self, model: Any, params: Dict[str, Any], optimizer: Any, step: int = 0
    ):
        self.model = model
        self.params = params
        self.optimizer = optimizer
        self.step = step
        self.opt_state = optimizer.init(params)

    def apply_gradients(self, grads: Any) -> "JAXTrainState":
        """Apply gradients and update state."""
        updates, new_opt_state = self.optimizer.update(
            grads, self.opt_state, self.params
        )
        new_params = jax.tree_map(lambda p, u: p + u, self.params, updates)
        return JAXTrainState(
            model=self.model,
            params=new_params,
            optimizer=self.optimizer,
            step=self.step + 1,
        )


# State management for PyTorch
class PyTorchTrainState:
    """Training state for PyTorch models."""

    def __init__(
        self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, step: int = 0
    ):
        self.model = model
        self.optimizer = optimizer
        self.step = step

    def apply_gradients(self) -> "PyTorchTrainState":
        """Apply gradients and update state."""
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.step += 1
        return self
