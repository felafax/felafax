# base.py
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union, Tuple

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase


@dataclass
class DatasetConfig:
    """Configuration for datasets."""

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
    mask_prompt: bool = False
    pad_id: int = 0

    # Optional transform function for custom preprocessing
    transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None


def load_data(
    config: DatasetConfig,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Loads the dataset and returns train and validation splits."""

    # Load dataset from Hugging Face Hub or local file
    if Path(config.data_source).is_file():
        dataset = load_dataset(
            "json",
            data_files=config.data_source,
            split=config.split,
        )
    else:
        dataset = load_dataset(
            config.data_source,
            split=config.split,
        )

    # If max_examples is set, limit the number of examples
    if config.max_examples is not None:
        dataset = dataset.select(range(min(config.max_examples, len(dataset))))

    # Split into train and validation sets
    dataset = dataset.train_test_split(
        test_size=config.train_test_split,
        seed=config.seed,
    )

    train_data = [sample for sample in dataset["train"]]
    val_data = [sample for sample in dataset["test"]]

    return train_data, val_data


def create_dataloader(
    config: DatasetConfig,
    dataset: "SFTDataset",
    shuffle: bool = False,
) -> DataLoader:
    """Creates a DataLoader for the given dataset."""
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        collate_fn=get_sft_collate_fn(
            max_seq_length=config.max_seq_length,
            pad_id=config.pad_id,
            ignore_index=config.ignore_index,
        ),
    )


class SFTDataset(Dataset):
    """Dataset for Supervised Fine-Tuning (SFT)."""

    def __init__(
        self,
        config: DatasetConfig,
        data: List[Dict[str, Any]],
        tokenizer: PreTrainedTokenizerBase,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.config = config

        self.max_seq_length = config.max_seq_length
        self.mask_prompt = config.mask_prompt
        self.ignore_index = config.ignore_index
        self.transform = config.transform
        self.eos_token_id = (
            self.tokenizer.eos_token_id
            if self.tokenizer.eos_token_id is not None
            else self.tokenizer.pad_token_id
        )

    def apply_format(self, example: Dict[str, Any]) -> Tuple[str, str]:
        """Default method to apply prompt formatting. Returns prompt and response.
        Override this method in subclasses for custom behavior."""
        prompt = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{example['instruction']}\n\n### Response:\n"
        )
        response = example["output"]
        return prompt, response

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        example = self.data[idx]

        if self.transform:
            example = self.transform(example)

        # Apply the prompt formatting
        prompt, response = self.apply_format(example)

        # Encode the prompt with special tokens
        encoded_prompt = self.tokenizer.encode(
            prompt,
            add_special_tokens=True,
            max_length=self.max_seq_length,
            truncation=True,
        )

        # Encode the response without adding special tokens
        encoded_response = self.tokenizer.encode(
            response,
            add_special_tokens=False,
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
        input_ids = torch.tensor(encoded_prompt_and_response, dtype=torch.long)

        # Create labels, masking the prompt if required
        labels = input_ids.clone()
        if self.mask_prompt:
            labels[: len(encoded_prompt)] = self.ignore_index

        return {
            "input_ids": input_ids,
            "labels": labels,
            "prompt_length": len(encoded_prompt),
            "response_length": len(encoded_response),
        }


def get_sft_collate_fn(
    max_seq_length: int = -1,
    pad_id: int = 0,
    ignore_index: int = -100,
) -> Callable:
    """Returns the collate function for supervised fine-tuning."""
    return partial(
        _sft_collate_fn,
        max_seq_length=max_seq_length,
        pad_id=pad_id,
        ignore_index=ignore_index,
    )


def _sft_collate_fn(
    samples: List[Dict[str, Any]],
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
