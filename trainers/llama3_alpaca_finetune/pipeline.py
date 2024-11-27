import os
import jax
from dotenv import load_dotenv
from transformers import AutoTokenizer
from src.felafax.trainer_engine.trainer import Trainer, TrainerConfig
from src.felafax.trainer_engine.setup import setup_environment
from src.felafax.trainer_engine.checkpoint import (
    Checkpointer,
    CheckpointerConfig,
)
from src.felafax.trainer_engine.data.data import (
    DatasetConfig,
    load_data,
    create_dataloader,
    SFTDataset,
)
from src.felafax.trainer_engine import utils


load_dotenv()
TEST_MODE = False
HF_TOKEN = os.getenv("HF_TOKEN") or input(
    "Please enter your HuggingFace token: "
)
BASE_DIR = os.getenv("BASE_DIR") or input(
    "Please enter the base directory for the training run: "
)

# Tip: To avoid entering these values manually, create a .env file in the `llama3_alpaca_finetune` folder with:
#   HF_TOKEN=your_huggingface_token
#   BASE_DIR=path_to_base_directory

########################################################
# Configure the dataset pipeline
########################################################
tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-3.2-1B", token=HF_TOKEN
)

dataset_config = DatasetConfig(
    data_source="yahma/alpaca-cleaned",
    max_seq_length=32,
    batch_size=8,
    num_workers=4,
    mask_prompt=False,
    train_test_split=0.15,
    # Setting max_examples limits the number of examples in the dataset.
    # This is useful for testing the pipeline without running the entire dataset.
    max_examples=100 if TEST_MODE else None,
    ignore_index=-100,
    pad_id=0,
    seed=42,
)

# Download and load the data files
train_data, val_data = load_data(config=dataset_config)

# Create datasets for SFT (supervised fine-tuning)
train_dataset = SFTDataset(
    config=dataset_config,
    data=train_data,
    tokenizer=tokenizer,
)
val_dataset = SFTDataset(
    config=dataset_config,
    data=val_data,
    tokenizer=tokenizer,
)

# Create dataloaders
train_dataloader = create_dataloader(
    config=dataset_config,
    dataset=train_dataset,
    shuffle=True,
)
val_dataloader = create_dataloader(
    config=dataset_config,
    dataset=val_dataset,
    shuffle=False,
)


########################################################
# Configure the trainer pipeline
########################################################
trainer_config = TrainerConfig(
    model_name="meta-llama/Llama-3.2-1B",
    param_dtype="bfloat16",
    compute_dtype="bfloat16",
    num_epochs=1,
    num_steps=5,
    num_tpus=jax.device_count(),
    lora_rank=16,
    use_lora=True,
    learning_rate=1e-3,
    base_dir=BASE_DIR,
    hf_token=HF_TOKEN,
    log_interval=1,
    eval_interval=50,
    eval_steps=5,
)

# Set up the training environment using trainer_config
setup_environment(trainer_config.base_dir)

# Configure the checkpointer
checkpoint_dir = f"{trainer_config.base_dir}/checkpoints/"
os.makedirs(checkpoint_dir, exist_ok=True)

checkpointer_config = CheckpointerConfig(
    checkpoint_dir=checkpoint_dir,
    max_to_keep=2,
    save_interval_steps=50,
    erase_existing_checkpoints=True,
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
trainer.train()

export_dir = f"{trainer_config.base_dir}/hf_export/"

# Run this to export the model in HF format
trainer.export(export_dir=export_dir)

# Run this to upload the exported model to HF
utils.upload_dir_to_hf(
    dir_path=export_dir,
    repo_name="felarof01/test-llama3-alpaca",
    token=HF_TOKEN,
)
