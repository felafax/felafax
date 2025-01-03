import os
import jax
from dotenv import load_dotenv
from transformers import AutoTokenizer
from src.felafax.trainer_engine.trainer import TrainerConfig
from src.felafax.trainer_engine.setup import setup_environment
from src.felafax.trainer_engine.checkpoint import (
    Checkpointer,
    CheckpointerConfig,
)
from .dataset import AlpacaDataset, AlpacaDatasetConfig
from src.felafax.trainer_engine import utils

########################################################
# The custom_trainer_example demonstrates the advantages of Felafax's component like design. You can easily extend any of the components.
# In this project, we customize the Trainer's optimizer to use consine schedule and weight decay.
########################################################
from .trainer import CustomTrainer  # Import your custom trainer

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN is None:
    HF_TOKEN = input(
        "Please input your HuggingFace token. Alternatively, you can create a .env file in the `llama3_alpaca_finetune` folder and specify HF_TOKEN there: "
    )
TEST_MODE = False

########################################################
# Configure the dataset pipeline
########################################################
tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-3.2-1B", token=HF_TOKEN
)
dataset_config = AlpacaDatasetConfig(
    data_source="yahma/alpaca-cleaned",
    max_seq_length=32,
    batch_size=8,
    num_workers=4,
    mask_prompt=False,
    train_test_split=0.15,
    # Setting max_examples limits the number of examples in the dataset.
    # This is useful for testing the pipeline without running the entire dataset.
    max_examples=100 if TEST_MODE else None,
    seed=42,
)
alpaca_dataset = AlpacaDataset(config=dataset_config)
alpaca_dataset.setup(tokenizer=tokenizer)

train_dataloader = alpaca_dataset.train_dataloader()
val_dataloader = alpaca_dataset.val_dataloader()


########################################################
# Configure the trainer pipeline
########################################################
trainer_config = TrainerConfig(
    model_name="meta-llama/Llama-3.2-1B",
    hf_token=HF_TOKEN,
    num_steps=20,
    num_tpus=jax.device_count(),
    base_dir="/Users/felarof99/Workspaces/GITHUB/building/checkpoints/",
    learning_rate=1e-4,  # You can customize other configurations as needed
    weight_decay=0.01,
    max_grad_norm=1.0,
)

# Set up the training environment using trainer_config
setup_environment(trainer_config.base_dir)

# Configure the checkpointer
checkpointer_config = CheckpointerConfig(
    checkpoint_dir=f"{trainer_config.base_dir}/checkpoints/",
)
checkpointer = Checkpointer(config=checkpointer_config)

# Initialize the custom trainer
trainer = CustomTrainer(
    trainer_config=trainer_config,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    checkpointer=checkpointer,
)

# Run training
trainer.train()

# Export the model in HF format (if needed)
# trainer.export()

# # Upload exported model to HF
# utils.upload_dir_to_hf(
#     dir_path=f"{trainer_config.base_dir}/hf_export/",
#     repo_name="felarof01/test-llama3-alpaca",
# )
