# base.py
from abc import abstractmethod, ABC
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset
from torch import Tensor
from felafax.prompts import PromptStyle


@dataclass
class BaseDataset(ABC):
    """Base class for all data modules in Felafax."""
    batch_size: int = 32
    max_seq_length: int = -1
    num_workers: int = 4
    ignore_index: int = -100
    prompt_style: Union[str, PromptStyle] = "alpaca"
    mask_prompt: bool = False

    @abstractmethod
    def setup(
        self,
        tokenizer: Optional[Any] = None,
    ) -> None:
        """Prepares the dataset for training by initializing tokenizer and loading data.

        This method must be called before accessing any dataloaders. It handles:
        - Setting up the tokenizer for text encoding
        - Loading the dataset (from HuggingFace Hub or local files) Creating train/validation datasets with proper tokenization

        Args:
            tokenizer: The tokenizer to use for encoding text. If None, uses previously set tokenizer.
        """
        pass


class SFTDataset(Dataset):
    """An in-memory dataset for supervised fine-tuning with `input_ids` and `labels`.

    Args:
        data: A list of samples (dicts). The target/label must be stored under the key 'output' and the instruction
            or other data can be stored under any key as long as it is compatible with the given prompt template.
        tokenizer: The tokenizer to use. Should match the one that was used to pretrain the model.
        prompt_style: The style to apply to prompts. See `felafax.trainer_engine.prompts` for a list of available styles.
        max_seq_length: Truncate sequences that are longer than this value. By default, no truncation is applied.
        mask_prompt: Whether to mask the prompt section from the label (with ``ignore_index``).
        ignore_index: The index to use for elements to be ignored in the label.
        transform: An optional transform to apply to the sample before it gets tokenized. Use this to rename the
            keys in the dataset to the expected 'instruction' and 'output' keys.

    Returns a dict with two keys:
        input_ids: The encoded prompt + response
        labels: Same as input_ids, unless ``mask_prompt=True`` in which case the 'prompt' part is replaced with
            the ``ignore_index``.
    """

    def __init__(
        self,
        data: List[Dict[str, str]],
        tokenizer: Any,
        prompt_style: Union[str, PromptStyle],
        max_seq_length: int = -1,
        mask_prompt: bool = True,
        ignore_index: int = -100,
        transform: Optional[Callable[[Any], Any]] = None,
    ) -> None:
        self.data = data
        self.tokenizer = tokenizer
        if isinstance(prompt_style, PromptStyle):
            self.prompt_style = prompt_style
        else:
            self.prompt_style = PromptStyle.from_name(prompt_style)
        self.max_seq_length = max_seq_length
        self.mask_prompt = mask_prompt
        self.ignore_index = ignore_index
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Union[Tensor, int]]:
        example = self.data[idx]

        # Apply any transform function to the example if provided.
        if self.transform is not None:
            example = self.transform(example)

        prompt = self.prompt_style.apply(prompt=example["instruction"],
                                         **example)

        # Encode the prompt with special tokens
        encoded_prompt = self.tokenizer.encode(
            prompt,
            add_special_tokens=True,
            max_length=self.max_seq_length,
            truncation=True,
        )

        # Encode the response with special tokens
        encoded_response = self.tokenizer.encode(
            example["output"],
            add_special_tokens=True,
            max_length=self.max_seq_length,
            truncation=True,
        )

        # Concatenate the encoded prompt and response
        encoded_prompt_and_response = encoded_prompt + encoded_response

        # Truncate the combined sequence to the max_seq_length if necessary
        if self.max_seq_length > 0:
            encoded_prompt_and_response = encoded_prompt_and_response[:self.
                                                                      max_seq_length]

        # Convert to torch tensor
        encoded_prompt_and_response = torch.tensor(encoded_prompt_and_response,
                                                   dtype=torch.int64)

        # Create labels, masking the prompt if required
        labels = encoded_prompt_and_response.clone()
        if self.mask_prompt:
            labels[:len(encoded_prompt)] = self.ignore_index

        # Calculate the total token count including the prompt
        total_token_count = len(encoded_prompt_and_response)

        return {
            "input_ids": encoded_prompt_and_response,
            "labels": labels,
            "token_count": total_token_count,
        }


def get_sft_collate_fn(max_seq_length: int = -1,
                       pad_id: int = 0,
                       ignore_index: int = -100):
    """Returns the collate function for supervised fine-tuning (needed in the DataLoader)."""
    return partial(
        _sft_collate_fn,
        max_seq_length=max_seq_length,
        pad_id=pad_id,
        ignore_index=ignore_index,
    )


def _sft_collate_fn(
    samples: List[Dict[str, Union[Tensor, int]]],
    max_seq_length: int = -1,
    pad_id: int = 0,
    ignore_index: int = -100,
) -> Dict[str, Tensor]:

    batched = {}
    for key in ("input_ids", "labels"):
        pad_value = pad_id if key == "input_ids" else ignore_index

        # Pad sequences to the longest sequence in the batch
        batched[key] = torch.nn.utils.rnn.pad_sequence(
            [sample[key] for sample in samples],
            batch_first=True,
            padding_value=pad_value,
        )

        # Truncate if needed
        if max_seq_length > 0:
            batched[key] = batched[key][:, :max_seq_length]

    # Collect token counts
    batched["token_count"] = torch.tensor(
        [sample["token_count"] for sample in samples],
        dtype=torch.int64).unsqueeze(1)

    return batched
