from dataclasses import dataclass, field
from pathlib import Path

from src.felafax.trainer_engine.trainer import TrainerConfig
from src.felafax.trainer_engine.data.data import DatasetConfig
from src.felafax.trainer_engine.checkpoint import CheckpointerConfig
import jax


@dataclass
class CLITunerDataConfig(DatasetConfig):
    """Configuration for MedQA dataset"""

    data_source: str = "ngram/medchat-qa"
    batch_size: int = 8

    max_seq_length: int = 4096
    num_workers: int = 8
    mask_prompt: bool = False

    dataset_input_field: str = "instruction"
    dataset_output_field: str = "output"


@dataclass
class CLITunerTrainerConfig(TrainerConfig):
    """Configuration for MedQA training"""

    model_name: str = "meta-llama/Llama-3.2-1B-Instruct"
    param_dtype: str = "bfloat16"
    compute_dtype: str = "bfloat16"
    num_epochs: int = 1
    num_steps: int = 100
    num_tpus: int = jax.device_count()

    learning_rate: float = 1e-3
    lora_rank: int = 16
    use_lora: bool = True

    log_interval: int = 5
    eval_interval: int = 5
    eval_steps: int = 10
    restore_checkpoint: bool = False

    hf_token: str = ""
    base_dir: str = "/mnt/persistent-disk"


@dataclass
class CLITunerCheckpointConfig(CheckpointerConfig):
    """Configuration for MedQA checkpointing"""

    checkpoint_dir: str = ""
    max_to_keep: int = 2
    save_interval_steps: int = 50
    enable_async_checkpointing: bool = True
    erase_existing_checkpoints: bool = True 


@dataclass
class PipelineConfig:
    """Main configuration for the MedQA training pipeline"""

    trainer_dir: str = ""
    export_dir: str = ""
    hf_token: str = ""
    hf_model_download_token: str = ""
    hf_repo: str = ""
    # token for downloading the model from Hugging Face
    hf_model_download_token: str = ""
    test_mode: bool = False

    data_config: CLITunerDataConfig = field(default_factory=CLITunerDataConfig)
    trainer_config: CLITunerTrainerConfig = field(
        default_factory=CLITunerTrainerConfig
    )
    checkpointer_config: CLITunerCheckpointConfig = field(
        default_factory=CLITunerCheckpointConfig
    )

    def __post_init__(self):
        trainer_dir = Path(self.trainer_dir)

        # TODO(cleanup): clean up base_dir.
        self.trainer_config.base_dir = str(trainer_dir)

        # When using CLI, use prepopulated hf download token for download model and tokenizer.
        self.trainer_config.hf_token = self.hf_model_download_token

        self.checkpointer_config.checkpoint_dir = str(
            trainer_dir / "checkpoints"
        )

        if isinstance(self.trainer_config.mesh_shape, list):
            self.trainer_config.mesh_shape = tuple(
                self.trainer_config.mesh_shape
            )
