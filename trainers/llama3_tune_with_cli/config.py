from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from src.felafax.trainer_engine.trainer import TrainerConfig
from src.felafax.trainer_engine.data.data import DatasetConfig
from src.felafax.trainer_engine.checkpoint import CheckpointerConfig


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
    base_dir: str = "/mnt/persistent-disk"
    param_dtype: str = "bfloat16"
    compute_dtype: str = "bfloat16"
    num_epochs: int = 1
    num_steps: int = 100
    mesh_shape: Tuple[int, int, int] = (1, 1, 4)
    learning_rate: float = 1e-3
    lora_rank: int = 16
    use_lora: bool = True
    log_interval: int = 5
    eval_interval: int = 5
    eval_steps: int = 10


@dataclass
class MedQACheckpointConfig(CheckpointerConfig):
    """Configuration for MedQA checkpointing"""
    max_to_keep: int = 2
    save_interval_steps: int = 50
    enable_async_checkpointing: bool = True


@dataclass
class PipelineConfig:
    """Main configuration for the MedQA training pipeline"""
    hf_token: str
    test_mode: bool = False
    model_repo: str = "felarof01/test-llama3.1-8b-medqa-finetuned-2048-no-mask"
    
    data: MedQADataConfig = MedQADataConfig()
    trainer: MedQATrainerConfig = MedQATrainerConfig()
    checkpoint: MedQACheckpointConfig = MedQACheckpointConfig()

    def __post_init__(self):
        # Update paths and tokens
        self.checkpoint.checkpoint_dir = str(self.base_dir / "checkpoints")
        self.trainer.base_dir = str(self.base_dir)
        self.trainer.hf_token = self.hf_token
