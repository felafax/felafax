import jax
from dotenv import load_dotenv
import os
from transformers import AutoTokenizer
from src.felafax.trainer_engine.trainer import Trainer, TrainerConfig
from src.felafax.trainer_engine.setup import setup_environment
from src.felafax.trainer_engine.checkpoint import (
    Checkpointer,
    CheckpointerConfig,
)
from src.felafax.trainer_engine.data.base import (
    DatasetConfig,
    load_data,
    create_dataloader,
    SFTDataset,
)
from src.felafax.trainer_engine import utils
from .dataset import create_med_qa_loaders, DatasetConfig


load_dotenv()
TEST_MODE = False
HF_TOKEN = os.getenv("HF_TOKEN") or input(
    "Please enter your HuggingFace token: "
)
BASE_DIR = os.getenv("BASE_DIR") or input(
    "Please enter the base directory for the training run: "
)
# Tip: To avoid entering these values manually, create a .env file in the `llama3_medqa` folder with:
#   HF_TOKEN=your_huggingface_token
#   BASE_DIR=path_to_base_directory

########################################################
# Configure the dataset pipeline
########################################################
# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-3.2-1B", token=HF_TOKEN
)

# Create dataset configuration for MedQA
medqa_config = DatasetConfig(
    data_source="ngram/medchat-qa",
    batch_size=32,
    max_seq_length=128,
    split="train",
)

# Create dataloaders for SFT
train_dataloader, val_dataloader = create_med_qa_loaders(
    config=medqa_config, tokenizer=tokenizer
)

########################################################
# Configure the trainer pipeline
########################################################
trainer_config = TrainerConfig(
    model_name="meta-llama/Llama-3.2-1B",
    hf_token=HF_TOKEN,
    num_steps=20,
    num_tpus=jax.device_count(),
    use_lora=True,
    lora_rank=8,
    learning_rate=1e-3,
    base_dir=BASE_DIR,
)

# Set up the training environment using trainer_config
setup_environment(trainer_config)

# Configure the checkpointer
checkpointer_config = CheckpointerConfig(
    checkpoint_dir=f"{trainer_config.base_dir}/checkpoints/",
)
checkpointer = Checkpointer(config=checkpointer_config)

# Put everything together and initialize the trainer
trainer = Trainer(
    trainer_config=trainer_config,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    checkpointer=checkpointer,
)

# Run training
# trainer.train()

export_dir = "felafax-storage/checkpoints/llama3_medqa_base/"

# Export the model in HF format
trainer.export(export_dir=export_dir)

# Upload exported model to HF
utils.upload_dir_to_hf(
    dir_path=export_dir,
    repo_name="felarof01/test-llama3-medqa-base",
    token=HF_TOKEN,
)
