import jax
import pyrallis
from pathlib import Path
from transformers import AutoTokenizer
from src.felafax.trainer_engine.trainer import Trainer
from src.felafax.trainer_engine.setup import setup_environment
from src.felafax.trainer_engine.checkpoint import Checkpointer
from src.felafax.trainer_engine import utils
from .dataset import create_dataloaders
from .config import PipelineConfig


@pyrallis.wrap()
def main(cfg: PipelineConfig):
    # Set up training environment first
    setup_environment(cfg.trainer)
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.trainer.model_name, 
        token=cfg.hf_token
    )

    # Create dataloaders
    train_dataloader, val_dataloader = create_dataloaders(
        config=cfg.data, 
        tokenizer=tokenizer
    )

    # Initialize checkpointer
    checkpointer = Checkpointer(cfg.checkpoint)

    # Initialize trainer
    trainer = Trainer(
        trainer_config=cfg.trainer,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        # checkpointer=checkpointer,
    )

    # Train the model
    trainer.train()

    # Export and upload model
    export_dir = Path(cfg.base_dir) / "hf_export"
    trainer.export(export_dir=str(export_dir))
    
    utils.upload_dir_to_hf(
        dir_path=str(export_dir),
        repo_name=cfg.model_repo,
        token=cfg.hf_token,
    )


if __name__ == "__main__":
    main()
    