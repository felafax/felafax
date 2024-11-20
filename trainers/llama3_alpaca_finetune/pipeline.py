import os
import jax
from dotenv import load_dotenv
from transformers import AutoTokenizer
from src.felafax.trainer_engine.trainer import Trainer, TrainerConfig
from src.felafax.trainer_engine.setup import setup_environment
from src.felafax.trainer_engine.checkpoint import Checkpointer, CheckpointerConfig
from src.felafax.trainer_engine.data.base import DefaultDatasetLoader, DatasetConfig
from src.felafax.trainer_engine import utils


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
dataset_config = DatasetConfig(
    data_source="yahma/alpaca-cleaned",
    max_seq_length=32,
    batch_size=8,
    num_workers=4,
    mask_prompt=False,
    train_test_split=0.15,
    prompt_style="alpaca",
    # Setting max_examples limits the number of examples in the dataset.
    # This is useful for testing the pipeline without running the entire dataset.
    max_examples=100 if TEST_MODE else None,
    seed=42,
)
dataset = DefaultDatasetLoader(config=dataset_config, tokenizer=tokenizer)
train_dataloader = dataset.train_dataloader()
val_dataloader = dataset.val_dataloader()


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
    base_dir="~/",
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

# Export the model in HF format
# trainer.export()

# # Upload exported model to HF
# utils.upload_dir_to_hf(
#     dir_path=f"{trainer_config.base_dir}/hf_export/",
#     repo_name="felarof01/test-llama3-alpaca",
# )
