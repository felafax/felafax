import jax
import pyrallis
from pathlib import Path
from transformers import AutoTokenizer
from src.felafax.trainer_engine.trainer import Trainer
from src.felafax.trainer_engine.setup import setup_environment, setup_distributed_jax
from src.felafax.trainer_engine.checkpoint import Checkpointer
from src.felafax.trainer_engine import utils
from .dataset import create_dataloaders
from .config import PipelineConfig


def main(cfg: PipelineConfig):
    # Set up training environment.
    setup_distributed_jax()
    setup_environment(cfg.trainer_config.base_dir)
    setup_distributed_jax()
    
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.trainer_config.model_name, token=cfg.trainer_config.hf_token
    )

    # Create dataloaders
    train_dataloader, val_dataloader = create_dataloaders(
        config=cfg.data_config, tokenizer=tokenizer
    )

    # Initialize checkpointer
    checkpointer = Checkpointer(cfg.checkpointer_config)

    # Initialize trainer
    trainer = Trainer(
        trainer_config=cfg.trainer_config,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        checkpointer=checkpointer,
    )

    # Train the model
    trainer.train()

    # Export and upload model
    trainer.export(export_dir=cfg.export_dir)

    utils.upload_dir_to_hf(
        dir_path=cfg.export_dir,
        repo_name=cfg.hf_repo,
        token=cfg.hf_token,
    )


if __name__ == "__main__":
    cfg = pyrallis.parse(config_class=PipelineConfig)
    main(cfg)
