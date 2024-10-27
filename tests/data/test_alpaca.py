import pytest
from transformers import AutoTokenizer
from typing import Tuple, Optional

from felafax.trainer_engine.data.alpaca import AlpacaDataModule
from felafax.prompts import PromptStyle


def setup_alpaca_data_module(batch_size: int, max_seq_length: int, num_workers: int = 2) -> Tuple[AutoTokenizer, AlpacaDataModule]:
    """
    Helper function to initialize the tokenizer and AlpacaDataModule.
    """
    # Download the tokenizer from Hugging Face for llama-3.1-8B
    tokenizer = AutoTokenizer.from_pretrained(
        "felafax/tokenizer-llama-3.1-8B-Instruct-JAX")
    tokenizer.pad_token = tokenizer.eos_token  # Ensure pad_token is set

    # Create the Alpaca Data Module with configuration
    data_module = AlpacaDataModule(
        batch_size=batch_size,
        max_seq_length=max_seq_length,
        num_workers=num_workers,
    )

    # Setup data
    data_module.setup(tokenizer=tokenizer)

    return tokenizer, data_module


def test_dataloader_length():
    tokenizer, data_module = setup_alpaca_data_module(batch_size=2, max_seq_length=10)
    train_dataloader = data_module.train_dataloader()

    # Test dataloader lengths
    assert len(train_dataloader) > 0, "Train dataloader is empty."


def test_batch_structure():
    tokenizer, data_module = setup_alpaca_data_module(batch_size=2, max_seq_length=10)
    train_dataloader = data_module.train_dataloader()
    train_batch = next(iter(train_dataloader))

    # Test batch keys
    expected_keys = {"input_ids", "labels", "token_counts"}
    assert set(
        train_batch.keys()
    ) == expected_keys, f"Train batch has unexpected keys: {train_batch.keys()}"


def test_batch_shapes():
    tokenizer, data_module = setup_alpaca_data_module(batch_size=2, max_seq_length=10)
    train_dataloader = data_module.train_dataloader()
    train_batch = next(iter(train_dataloader))

    # Test batch shapes
    for key in ["input_ids", "labels"]:
        assert train_batch[key].shape[
            0] == data_module.batch_size, f"Unexpected batch size for train_batch[{key}]"
        assert train_batch[key].shape[
            1] <= data_module.max_seq_length, f"Sequence length exceeds max_seq_length for {key}"


def test_prompt_style():
    tokenizer, data_module = setup_alpaca_data_module(batch_size=2, max_seq_length=10)
    train_dataloader = data_module.train_dataloader()

    # Test dataset attributes
    assert isinstance(train_dataloader.dataset.prompt_style,
                      PromptStyle), "Prompt style is not set correctly."


def test_special_tokens_with_large_seq_length():
    tokenizer, data_module = setup_alpaca_data_module(batch_size=2, max_seq_length=128*1024)
    train_dataloader = data_module.train_dataloader()
    train_batch = next(iter(train_dataloader))

    # Test for special tokens when truncation is unlikely
    sample_ids = train_batch["input_ids"][0]  # Get first sequence in batch
    assert tokenizer.bos_token_id in sample_ids, "BOS token not found in sequence"
    assert tokenizer.eos_token_id in sample_ids, "EOS token not found in sequence"
