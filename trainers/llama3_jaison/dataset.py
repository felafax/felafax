from dataclasses import dataclass
from src.felafax.trainer_engine.data.data import (
    SFTDataset,
    DatasetConfig,
    load_data,
    create_dataloader,
)
from typing import Dict, Any, Tuple
from transformers import PreTrainedTokenizerBase


class CustomDataset(SFTDataset):
    """Custom dataset for MedQA data."""

    def apply_format(self, example: Dict[str, Any]) -> Tuple[str, str]:
        """Override apply_format to provide custom prompt formatting."""
        input_prompt = example[self.config.dataset_input_field]
        response_prompt = example[self.config.dataset_output_field]

        prompt = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{input_prompt}\n\n### Response:\n"
        )
        return prompt, response_prompt


def create_dataloaders(
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
    train_dataset = CustomDataset(
        config=config,
        data=train_data,
        tokenizer=tokenizer,
    )
    val_dataset = CustomDataset(
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
