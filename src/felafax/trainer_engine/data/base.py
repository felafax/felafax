# base.py
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase

from src.felafax.trainer_engine.data.prompts import BasePromptTemplate


@dataclass
class DatasetConfig:
    """Base configuration for datasets."""

    # Data loading parameters
    data_source: str = ""
    max_examples: Optional[int] = None
    split: str = "train"
    train_test_split: float = 0.15
    seed: int = 42

    # Processing parameters
    batch_size: int = 32
    max_seq_length: int = 64
    num_workers: int = 4
    ignore_index: int = -100
    prompt_style: Union[str, BasePromptTemplate] = "alpaca"
    mask_prompt: bool = False
    pad_id: int = 0

    # Optional transform function for custom preprocessing
    transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None


class DefaultDatasetLoader:
    """Base class for datasets in Felafax."""

    def __init__(
        self, config: DatasetConfig, tokenizer: PreTrainedTokenizerBase
    ):
        self.config = config
        self.tokenizer = tokenizer

        if isinstance(self.config.prompt_style, str):
            self.prompt_template = BasePromptTemplate.from_name(
                self.config.prompt_style
            )
        else:
            self.prompt_template = self.config.prompt_style

        self.train_dataset = None
        self.val_dataset = None
        self.setup()

    def setup(self) -> None:
        """Sets up the dataset by loading data and creating train/validation splits."""

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

        # If max_examples is set, limit the number of examples
        if self.config.max_examples is not None:
            dataset = dataset.select(
                range(min(self.config.max_examples, len(dataset)))
            )

        # Split into train and validation sets
        dataset = dataset.train_test_split(
            test_size=self.config.train_test_split, seed=self.config.seed
        )

        self.train_data = [sample for sample in dataset["train"]]
        self.val_data = [sample for sample in dataset["test"]]

    def train_dataloader(self) -> DataLoader:
        self.train_dataset = SFTDataset(
            data=self.train_data,
            tokenizer=self.tokenizer,
            prompt_template=self.prompt_template,
            max_seq_length=self.config.max_seq_length,
            mask_prompt=self.config.mask_prompt,
            ignore_index=self.config.ignore_index,
            transform=self.config.transform,
        )

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
        self.val_dataset = SFTDataset(
            data=self.val_data,
            tokenizer=self.tokenizer,
            prompt_template=self.prompt_template,
            max_seq_length=self.config.max_seq_length,
            mask_prompt=self.config.mask_prompt,
            ignore_index=self.config.ignore_index,
            transform=self.config.transform,
        )

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


class SFTDataset(Dataset):
    """Creates a dataset for supervised fine-tuning."""

    def __init__(
        self,
        data: List[Dict[str, str]],
        tokenizer: Any,
        prompt_template: Union[str, BasePromptTemplate],
        max_seq_length: int = -1,
        mask_prompt: bool = True,
        ignore_index: int = -100,
        transform: Optional[Callable[[Any], Any]] = None,
    ) -> None:
        self.data = data
        self.tokenizer = tokenizer
        self.prompt_template = prompt_template
        self.max_seq_length = max_seq_length
        self.mask_prompt = mask_prompt
        self.ignore_index = ignore_index
        self.transform = transform
        self.eos_token_id = (
            self.tokenizer.eos_token_id
            if self.tokenizer.eos_token_id is not None
            else self.tokenizer.pad_token_id
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, int]]:
        example = self.data[idx]

        # Apply any transform function to the example if provided
        if self.transform is not None:
            example = self.transform(example)

        prompt = self.prompt_template.apply(
            prompt=example.get("instruction", ""), **example
        )

        # Encode the prompt with special tokens
        encoded_prompt = self.tokenizer.encode(
            prompt,
            add_special_tokens=True,
            max_length=self.max_seq_length,
            truncation=True,
        )

        # Encode the response with special tokens
        encoded_response = self.tokenizer.encode(
            example.get("output", ""),
            add_special_tokens=True,
            max_length=self.max_seq_length,
            truncation=True,
        )

        # Concatenate the encoded prompt and response
        encoded_prompt_and_response = (
            encoded_prompt + encoded_response + [self.eos_token_id]
        )

        # Truncate the combined sequence to the max_seq_length if necessary
        if self.max_seq_length > 0:
            encoded_prompt_and_response = encoded_prompt_and_response[
                : self.max_seq_length
            ]

        # Convert to torch tensor
        encoded_prompt_and_response = torch.tensor(
            encoded_prompt_and_response, dtype=torch.int64
        )

        # Create labels, masking the prompt if required
        labels = encoded_prompt_and_response.clone()
        if self.mask_prompt:
            labels[: len(encoded_prompt)] = self.ignore_index

        return {
            "input_ids": encoded_prompt_and_response,
            "labels": labels,
            "prompt_length": len(encoded_prompt),
            "response_length": len(encoded_response),
        }


def get_sft_collate_fn(
    max_seq_length: int = -1, pad_id: int = 0, ignore_index: int = -100
):
    """Returns the collate function for supervised fine-tuning."""
    return partial(
        _sft_collate_fn,
        max_seq_length=max_seq_length,
        pad_id=pad_id,
        ignore_index=ignore_index,
    )


def _sft_collate_fn(
    samples: List[Dict[str, Union[torch.Tensor, int]]],
    max_seq_length: int,
    pad_id: int = 0,
    ignore_index: int = -100,
) -> Dict[str, torch.Tensor]:
    """Collate function that pads sequences to max_seq_length."""
    batched = {}
    for key in ("input_ids", "labels"):
        pad_value = pad_id if key == "input_ids" else ignore_index

        # Truncate and pad sequences
        sequences = [sample[key][:max_seq_length] for sample in samples]
        padded_sequences = [
            torch.nn.functional.pad(
                seq, (0, max_seq_length - len(seq)), value=pad_value
            )
            for seq in sequences
        ]
        batched[key] = torch.stack(padded_sequences)

    # Process lengths
    for key in ("prompt_length", "response_length"):
        lengths = torch.tensor(
            [min(sample[key], max_seq_length) for sample in samples],
            dtype=torch.int64,
        ).unsqueeze(1)

        batched[key] = lengths

    return batched
