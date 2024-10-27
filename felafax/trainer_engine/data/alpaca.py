from dataclasses import dataclass
from typing import Optional, Any
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from pathlib import Path

from felafax.trainer_engine.data.base import DataModule, SFTDataset, get_sft_collate_fn
from felafax.prompts import PromptStyle

@dataclass
class AlpacaDataModule(DataModule):
    """Alpaca data module for supervised fine-tuning."""
    # Alpaca-specific fields
    data_source: str = "yahma/alpaca-cleaned"
    max_examples: Optional[int] = None
    split: str = "train"
    train_test_split: float = 0.15
    ignore_index: int = -100
    seed: int = 42
    
    def __post_init__(self):
        if isinstance(self.prompt_style, str):
            self.prompt_style = PromptStyle.from_name(self.prompt_style)
        self.tokenizer = None
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, tokenizer: Optional[Any] = None) -> None:
        self.tokenizer = tokenizer or self.tokenizer

        # Load dataset from Hugging Face Hub or local file
        if Path(self.data_source).is_file():
            dataset = load_dataset("json", data_files=self.data_source, split=self.split)
        else:
            dataset = load_dataset(self.data_source, split=self.split)

        # Limit number of examples
        if self.max_examples is not None:
            dataset = dataset.select(range(min(self.max_examples, len(dataset))))

        # Split into train and validation sets
        dataset = dataset.train_test_split(test_size=self.train_test_split, seed=self.seed)
        train_data = [sample for sample in dataset["train"]]
        val_data = [sample for sample in dataset["test"]]

        # Create datasets
        self.train_dataset = SFTDataset(
            data=train_data,
            tokenizer=self.tokenizer,
            prompt_style=self.prompt_style,
            max_seq_length=self.max_seq_length,
            mask_prompt=self.mask_prompt,
            ignore_index=self.ignore_index,
        )

        self.val_dataset = SFTDataset(
            data=val_data,
            tokenizer=self.tokenizer,
            prompt_style=self.prompt_style,
            max_seq_length=self.max_seq_length,
            mask_prompt=self.mask_prompt,
            ignore_index=self.ignore_index,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.seed),
            num_workers=self.num_workers,
            collate_fn=get_sft_collate_fn(
                max_seq_length=self.max_seq_length,
                ignore_index=self.ignore_index,
            ),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=get_sft_collate_fn(
                max_seq_length=self.max_seq_length,
                ignore_index=self.ignore_index,
            ),
        )
