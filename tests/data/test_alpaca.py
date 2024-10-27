import pytest
from transformers import AutoTokenizer
from dataclasses import dataclass
from typing import Optional

from felafax.trainer_engine.data.alpaca import AlpacaDataModule
from felafax.prompts import PromptStyle


def test_alpaca_data_module():
    # Download the tokenizer from Hugging Face for llama-3.1-8B
    tokenizer = AutoTokenizer.from_pretrained(
        "felafax/tokenizer-llama-3.1-8B-Instruct-JAX")
    tokenizer.pad_token = tokenizer.eos_token  # Ensure pad_token is set

    # Create the Alpaca Data Module with configuration
    data_module = AlpacaDataModule(
        batch_size=2,
        max_seq_length=10,
        num_workers=2,
    )

    # Setup data
    data_module.setup(tokenizer=tokenizer)

    # Get dataloaders
    train_dataloader = data_module.train_dataloader()

    # Test dataloader lengths
    assert len(train_dataloader) > 0, "Train dataloader is empty."

    # Test batch structure
    train_batch = next(iter(train_dataloader))

    # Test batch keys
    expected_keys = {"input_ids", "labels", "token_counts"}
    assert set(
        train_batch.keys()
    ) == expected_keys, f"Train batch has unexpected keys: {train_batch.keys()}"

    # Test batch shapes
    for key in ["input_ids", "labels"]:
        assert train_batch[key].shape[
            0] == data_module.batch_size, f"Unexpected batch size for train_batch[{key}]"
        assert train_batch[key].shape[
            1] <= data_module.max_seq_length, f"Sequence length exceeds max_seq_length for {key}"

    # Test dataset attributes
    assert isinstance(train_dataloader.dataset.prompt_style,
                      PromptStyle), "Prompt style is not set correctly."
