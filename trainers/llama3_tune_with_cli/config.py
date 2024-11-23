from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

from src.felafax.trainer_engine.trainer import TrainerConfig
from src.felafax.trainer_engine.data.data import DatasetConfig
from src.felafax.trainer_engine.checkpoint import CheckpointerConfig
import jax


@dataclass
class MedQADataConfig(DatasetConfig):
    """Configuration for MedQA dataset"""
    data_source: str = "ngram/medchat-qa"
    batch_size: int = 8
    max_seq_length: int = 4096
    num_workers: int = 8
    mask_prompt: bool = False


@dataclass
class MedQATrainerConfig(TrainerConfig):
    """Configuration for MedQA training"""
    model_name: str = "meta-llama/Llama-3.2-1B-Instruct"
    param_dtype: str = "bfloat16"
    compute_dtype: str = "bfloat16"
    num_epochs: int = 1
    num_steps: int = 100
    num_tpus: int = jax.device_count()
    mesh_shape: Tuple[int, int, int] = (1, 1, 4)
    learning_rate: float = 1e-3
    lora_rank: int = 16
    use_lora: bool = True
    log_interval: int = 5
    eval_interval: int = 5
    eval_steps: int = 10
    restore_checkpoint: bool = False


@dataclass
class MedQACheckpointConfig(CheckpointerConfig):
    """Configuration for MedQA checkpointing"""
    checkpoint_dir: str = ""
    max_to_keep: int = 2
    save_interval_steps: int = 50
    enable_async_checkpointing: bool = True


@dataclass
class PipelineConfig:
    """Main configuration for the MedQA training pipeline"""
    hf_token: str
    base_dir: str = "/mnt/persistent-disk"
    test_mode: bool = False
    model_repo: str = "felarof01/test-llama3.1-8b-medqa-finetuned-2048-no-mask"
    
    data: MedQADataConfig = field(default_factory=MedQADataConfig)
    trainer: MedQATrainerConfig = field(default_factory=MedQATrainerConfig)
    checkpoint: MedQACheckpointConfig = field(default_factory=MedQACheckpointConfig)

    def __post_init__(self):
        # Convert base_dir to Path for operations
        base_dir = Path(self.base_dir)
        # Update paths and tokens
        self.checkpoint.checkpoint_dir = str(base_dir / "checkpoints")
        self.trainer.base_dir = str(base_dir)
        self.trainer.hf_token = self.hf_token

        # Convert mesh_shape from list to tuple if it's a list
        if isinstance(self.trainer.mesh_shape, list):
            self.trainer.mesh_shape = tuple(self.trainer.mesh_shape)
