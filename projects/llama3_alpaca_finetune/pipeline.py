from transformers import AutoTokenizer
from felafax.trainer_engine.trainer import Trainer, TrainerConfig
from felafax.trainer_engine.setup import setup_environment
from felafax.trainer_engine.checkpoint import Checkpointer, CheckpointerConfig
from .dataset import AlpacaDataset, AlpacaDatasetConfig

########################################################
# Configure the dataset pipeline
########################################################
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
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
alpaca_dataset = AlpacaDataset(config=dataset_config)
alpaca_dataset.setup(tokenizer=tokenizer)

train_dataloader = alpaca_dataset.train_dataloader()
val_dataloader = alpaca_dataset.val_dataloader()


########################################################
# Configure the trainer pipeline
########################################################
trainer_config = TrainerConfig(
    model_name="meta-llama/Llama-2-7b-hf",
    num_steps=10,  # Adjust the number of training steps
    num_tpus=4,  # Adjust based on available TPUs
    base_dir="/mnt/persistent-disk",
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
trainer.train()
