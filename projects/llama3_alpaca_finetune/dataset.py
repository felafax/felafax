from dataclasses import dataclass
from typing import Optional, Any, Union
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset

from felafax.trainer_engine.data.base import (
    BaseDataset,
    SFTDataset,
    get_sft_collate_fn,
    DatasetConfig,
)
from felafax.trainer_engine.data.prompts import BasePromptTemplate


@dataclass
class AlpacaDatasetConfig(DatasetConfig):
    """Configuration for Alpaca dataset."""

    data_source: str = "yahma/alpaca-cleaned"
    max_seq_length: int = 64
    prompt_style: Union[str, BasePromptTemplate] = "alpaca"


class AlpacaDataset(BaseDataset):
    """Alpaca dataset for supervised fine-tuning."""

    def __init__(self, config: AlpacaDatasetConfig):
        super().__init__(config)

    def setup(self, tokenizer: Optional[Any] = None) -> None:
        self.tokenizer = tokenizer or self.tokenizer

        # Load dataset from Hugging Face Hub or local file
        if Path(self.config.data_source).is_file():
            dataset = load_dataset(
                "json",
                data_files=self.config.data_source,
                split=self.config.split,
            )
        else:
            dataset = load_dataset(
                self.config.data_source, split=self.config.split
            )

        # If max_examples is set, limit the number of examples.
        # Typically used for quick testing, so that you don't wait on loading
        # and mapping the entire dataset.
        if self.config.max_examples is not None:
            dataset = dataset.select(
                range(min(self.config.max_examples, len(dataset)))
            )

        # Split into train and validation sets
        dataset = dataset.train_test_split(
            test_size=self.config.train_test_split, seed=self.config.seed
        )

        train_data = [sample for sample in dataset["train"]]
        val_data = [sample for sample in dataset["test"]]

        # Create datasets
        self.train_dataset = SFTDataset(
            data=train_data,
            tokenizer=self.tokenizer,
            prompt_template=self.config.prompt_style,
            max_seq_length=self.config.max_seq_length,
            mask_prompt=self.config.mask_prompt,
            ignore_index=self.config.ignore_index,
        )

        self.val_dataset = SFTDataset(
            data=val_data,
            tokenizer=self.tokenizer,
            prompt_template=self.config.prompt_style,
            max_seq_length=self.config.max_seq_length,
            mask_prompt=self.config.mask_prompt,
            ignore_index=self.config.ignore_index,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.config.seed),
            num_workers=self.config.num_workers,
            collate_fn=get_sft_collate_fn(
                max_seq_length=self.config.max_seq_length,
                pad_id=self.config.pad_id,
                ignore_index=self.config.ignore_index,
            ),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=get_sft_collate_fn(
                max_seq_length=self.config.max_seq_length,
                pad_id=self.config.pad_id,
                ignore_index=self.config.ignore_index,
            ),
        )
