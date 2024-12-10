from dataclasses import dataclass
from src.felafax.trainer_engine.data.data import (
    SFTDataset,
    DatasetConfig,
    load_data,
    create_dataloader,
)
from typing import Dict, Any, Tuple
from transformers import PreTrainedTokenizerBase


class MedQADataset(SFTDataset):
    """Custom dataset for MedQA data."""

    def apply_format(self, example: Dict[str, Any]) -> Tuple[str, str]:
        """Override apply_format to handle MedQA format."""
        prompt = (
            "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. ",
            "### Instruction: Pretend you are a real estate agent. \n\n",
            f"### Input: {example['input']}\n\n",
            "### Response: \n\n",
        )
        response = example['response']
        return prompt, response


def create_med_qa_loaders(
    config: DatasetConfig, tokenizer: PreTrainedTokenizerBase
):
    """Creates and returns train and validation dataloaders for MedQA dataset.

    Args:
        config: DatasetConfig object with dataset parameters
        tokenizer: Pre-initialized tokenizer

    Returns:
        Tuple containing train_dataloader and val_dataloader
    """

    # Load the data
    train_data, val_data = load_data(config=config)

    # Create datasets
    train_dataset = MedQADataset(
        config=config,
        data=train_data,
        tokenizer=tokenizer,
    )
    val_dataset = MedQADataset(
        config=config,
        data=val_data,
        tokenizer=tokenizer,
    )

    # Create and return dataloaders
    train_dataloader = create_dataloader(
        config=config,
        dataset=train_dataset,
        shuffle=True,
    )
    val_dataloader = create_dataloader(
        config=config,
        dataset=val_dataset,
        shuffle=False,
    )

    return train_dataloader, val_dataloader
