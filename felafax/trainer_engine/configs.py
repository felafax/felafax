from dataclasses import dataclass, asdict
from typing import Literal
import json
from pathlib import Path


@dataclass
class ModelConfig:
    model_name: str
    use_lora: bool
    lora_rank: int
    lora_alpha: int
    lora_dropout: float


@dataclass
class TrainingConfig:
    learning_rate: float
    num_epochs: int
    batch_size: int
    max_steps: int
    gradient_accumulation_steps: int
    seq_length: int
    print_every_n_steps: int
    eval_every_n_steps: int
    max_eval_steps: int


@dataclass
class DistributedConfig:
    backend: Literal["jax", "pytorch_xla"]
    dtype: str
    param_dtype: str


@dataclass
class OptimizerConfig:
    name: str
    weight_decay: float
    max_grad_norm: float
    warmup_steps: int
    lr_scheduler: str


@dataclass
class LoggingConfig:
    wandb_project: str
    log_every_n_steps: int
    save_every_n_steps: int
    eval_every_n_steps: int


@dataclass
class WorkflowConfig:
    model: ModelConfig
    training: TrainingConfig
    distributed: DistributedConfig
    optimizer: OptimizerConfig
    logging: LoggingConfig
    name: str

    @classmethod
    def from_dict(cls, config_dict: dict) -> "WorkflowConfig":
        return WorkflowConfig(
            name=config_dict["name"],
            model=ModelConfig(**config_dict["model"]),
            training=TrainingConfig(**config_dict["training"]),
            distributed=DistributedConfig(**config_dict["distributed"]),
            optimizer=OptimizerConfig(**config_dict["optimizer"]),
            logging=LoggingConfig(**config_dict["logging"]),
        )

    @classmethod
    def from_json(cls, json_path: str | Path) -> "WorkflowConfig":
        with open(json_path, "r") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    def to_dict(self) -> dict:
        return {
            "model": asdict(self.model),
            "training": asdict(self.training),
            "distributed": asdict(self.distributed),
            "optimizer": asdict(self.optimizer),
            "logging": asdict(self.logging),
        }

    def to_json(self, json_path: str | Path) -> None:
        with open(json_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def __post_init__(self):
        # Add any validation logic here
        if self.distributed.backend not in ["jax", "pytorch_xla"]:
            raise ValueError("Backend must be either 'jax' or 'pytorch_xla'")
