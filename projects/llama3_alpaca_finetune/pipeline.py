from transformers import AutoTokenizer
from felafax.trainer_engine.trainer import Trainer, TrainerConfig
from felafax.trainer_engine.checkpoint import Checkpointer, CheckpointerConfig
from .dataset import AlpacaDataset, AlpacaDatasetConfig

# Instantiate the tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# Instantiate the dataset configuration
dataset_config = AlpacaDatasetConfig(
    data_source="yahma/alpaca-cleaned",
    max_seq_length=32,
    batch_size=8,
    num_workers=4,
    mask_prompt=False,
    train_test_split=0.15,
    max_examples=None,  # Set to an integer to limit examples
    seed=42,
)

# Set up the Alpaca dataset
alpaca_dataset = AlpacaDataset(config=dataset_config)
alpaca_dataset.setup(tokenizer=tokenizer)

# Get data loaders
train_dataloader = alpaca_dataset.train_dataloader()
val_dataloader = alpaca_dataset.val_dataloader()

# Instantiate the checkpointer configuration
checkpointer_config = CheckpointerConfig(
    checkpoint_dir="/home/checkpoints/",
)
checkpointer = Checkpointer(config=checkpointer_config)

# Instantiate the trainer configuration
trainer_config = TrainerConfig(
    model_name="meta-llama/Llama-2-7b-hf",
    num_steps=10,  # Adjust the number of training steps
    num_tpus=4,    # Adjust based on available TPUs
    checkpointer_config=checkpointer_config,
)

# Initialize the Trainer
trainer = Trainer(
    trainer_config=trainer_config,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    checkpointer=checkpointer,
)

# Run training
trainer.train()
