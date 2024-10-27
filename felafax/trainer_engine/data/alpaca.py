from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict, Any

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from pathlib import Path

from felafax.trainer_engine.data.base import DataModule, SFTDataset, get_sft_collate_fn
from felafax.prompts import PromptStyle


@dataclass
class AlpacaConfig:
    """Configuration for Alpaca dataset."""
    batch_size: int = 32
    max_seq_length: int = -1
    max_examples: Optional[int] = None
    data_source: str = "yahma/alpaca-cleaned"
    prompt_style: Union[str, PromptStyle] = "alpaca"
    mask_prompt: bool = False
    num_workers: int = 4
    split: str = "train"
    val_split_fraction: float = 0.15
    ignore_index: int = -100
    seed: int = 42

class AlpacaDataModule(DataModule):
    """Alpaca data module for supervised fine-tuning."""

    def __init__(self, config: Optional[AlpacaConfig] = None):
        self.config = config or AlpacaConfig()
        super().__init__()
        if isinstance(self.config.prompt_style, str):
            self.config.prompt_style = PromptStyle.from_name(
                self.config.prompt_style)
        self.tokenizer = None
        self.train_dataset = None
        self.val_dataset = None

    def connect(
        self,
        tokenizer: Optional[Any] = None,
        batch_size: Optional[int] = None,
        max_seq_length: Optional[int] = None,
    ) -> None:
        self.tokenizer = tokenizer or self.tokenizer
        if batch_size:
            self.config.batch_size = batch_size
        if max_seq_length:
            self.config.max_seq_length = max_seq_length

    def prepare_data(self) -> None:
        # No need to download data ahead of time; datasets library handles it
        pass

    def setup(self, stage: str = "") -> None:
        # Load dataset from Hugging Face Hub or local file
        if Path(self.config.data_source).is_file():
            dataset = load_dataset(
                "json",
                data_files=self.config.data_source,
                split=self.config.split,
            )
        else:
            dataset = load_dataset(self.config.data_source,
                                   split=self.config.split)

        # Limit number of examples
        if self.config.max_examples is not None:
            dataset = dataset.select(
                range(min(self.config.max_examples, len(dataset))))

        # Split into train and validation sets
        dataset = dataset.train_test_split(
            test_size=self.config.val_split_fraction, seed=self.config.seed)
        train_data = [sample for sample in dataset["train"]]
        val_data = [sample for sample in dataset["test"]]

        # Create datasets
        self.train_dataset = SFTDataset(
            data=train_data,
            tokenizer=self.tokenizer,
            prompt_style=self.config.prompt_style,
            max_seq_length=self.config.max_seq_length,
            mask_prompt=self.config.mask_prompt,
            ignore_index=self.config.ignore_index,
        )

        self.val_dataset = SFTDataset(
            data=val_data,
            tokenizer=self.tokenizer,
            prompt_style=self.config.prompt_style,
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
                ignore_index=self.config.ignore_index,
            ),
        )
