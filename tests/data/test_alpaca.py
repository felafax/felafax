"""Tests for the alpaca.py module.

pytest --capture=no tests/data/test_alpaca.py
"""

from transformers import AutoTokenizer
from typing import Tuple

from felafax.trainer_engine.data.alpaca import AlpacaDataset
from felafax.trainer_engine.data.prompts import PromptStyle


def setup_alpaca_dataset(
    batch_size: int, max_seq_length: int, num_workers: int = 2
) -> Tuple[AutoTokenizer, AlpacaDataset]:
    """Sets up the tokenizer and AlpacaDataset."""
    # Download the tokenizer from Hugging Face for llama-3.1-8B
    tokenizer = AutoTokenizer.from_pretrained(
        "felafax/tokenizer-llama-3.1-8B-Instruct-JAX"
    )
    data_module = AlpacaDataset(
        batch_size=batch_size,
        max_seq_length=max_seq_length,
        num_workers=num_workers,
    )

    # Setup data
    data_module.setup(tokenizer=tokenizer)

    return tokenizer, data_module


def test_dataloader_length():
    """Tests that the dataloader is not empty."""
    _, data_module = setup_alpaca_dataset(batch_size=2, max_seq_length=10)
    train_dataloader = data_module.train_dataloader()
    assert len(train_dataloader) > 0, "Train dataloader is empty."


def test_batch_keys():
    """Tests that dataset batch contains expected keys."""
    _, data_module = setup_alpaca_dataset(batch_size=2, max_seq_length=10)
    train_dataloader = data_module.train_dataloader()
    train_batch = next(iter(train_dataloader))

    expected_keys = {"input_ids", "labels", "token_count"}
    assert (
        set(train_batch.keys()) == expected_keys
    ), f"Train batch has unexpected keys: {train_batch.keys()}"


def test_batch_shapes():
    """Tests that the batch size and max_seq_length are being obeyed."""
    _, data_module = setup_alpaca_dataset(batch_size=2, max_seq_length=10)
    train_dataloader = data_module.train_dataloader()
    train_batch = next(iter(train_dataloader))

    for key in ["input_ids", "labels"]:
        assert (
            train_batch[key].shape[0] == data_module.batch_size
        ), f"Unexpected batch size for train_batch[{key}]"
        assert (
            train_batch[key].shape[1] <= data_module.max_seq_length
        ), f"Sequence length exceeds max_seq_length for {key}"


def test_prompt_style():
    """Tests that the prompt style is being applied correctly."""
    _, data_module = setup_alpaca_dataset(batch_size=2, max_seq_length=10)
    train_dataloader = data_module.train_dataloader()
    assert isinstance(
        train_dataloader.dataset.prompt_style, PromptStyle
    ), "Prompt style is not set correctly."


def test_special_tokens_with_large_seq_length():
    """Tests that the special tokens are being added correctly."""
    # Use a large sequence length to ensure there is at least one example that has EOS token.
    tokenizer, data_module = setup_alpaca_dataset(
        batch_size=1, max_seq_length=128 * 1024
    )
    train_dataloader = data_module.train_dataloader()
    train_batch = next(iter(train_dataloader))

    # Test for special tokens when truncation is unlikely
    sample_ids = train_batch["input_ids"][0]  # Get first sequence in batch
    assert (
        tokenizer.bos_token_id in sample_ids
    ), "BOS token not found in sequence"
    # TODO(ntnsonti): EOS token is not getting added, take a look.
    # assert (tokenizer.eos_token_id
    #         in sample_ids), "EOS token not found in sequence"
