"""Tests for the alpaca dataset implementation in pipeline.py"""

import pytest
import numpy as np
from transformers import AutoTokenizer

from .alpaca import AlpacaDataset, AlpacaDatasetConfig


@pytest.fixture
def tokenizer():
    """Fixture to provide a tokenizer for tests."""
    return AutoTokenizer.from_pretrained(
        "felafax/tokenizer-llama-3.1-8B-Instruct-JAX"
    )


@pytest.fixture
def dataset_config():
    """Fixture to provide dataset configuration for tests."""
    return AlpacaDatasetConfig(
        batch_size=2,
        max_seq_length=2048,
        data_source="yahma/alpaca-cleaned",
        max_examples=10,  # Limit examples for testing
    )


@pytest.fixture
def dataset(tokenizer, dataset_config):
    """Fixture to provide a configured dataset for tests."""
    dataset = AlpacaDataset(config=dataset_config)
    dataset.setup(tokenizer=tokenizer)
    return dataset


def test_setup(dataset):
    """Tests that dataset correctly initializes with basic config."""
    assert dataset.config.batch_size == 2
    assert dataset.config.max_seq_length == 2048
    assert hasattr(dataset, "train_dataset")
    assert hasattr(dataset, "val_dataset")


def test_output_keys_shapes(dataset):
    """Tests that dataloader produces correctly formatted batches."""
    train_loader = dataset.train_dataloader()
    batch = next(iter(train_loader))

    # Check batch keys
    assert "input_ids" in batch
    assert "labels" in batch

    # Check shapes
    assert batch["input_ids"].shape[0] == dataset.config.batch_size
    assert batch["labels"].shape[0] == dataset.config.batch_size
    assert batch["input_ids"].shape[1] <= dataset.config.max_seq_length
    assert batch["labels"].shape[1] <= dataset.config.max_seq_length


def test_labels_match_inputs(dataset):
    """Tests that labels match input_ids exactly."""
    # The token shifting for next-token prediction is handled by the trainer, not the dataset.
    train_loader = dataset.train_dataloader()
    batch = next(iter(train_loader))

    # Get first sequence in batch and convert to numpy
    input_ids = batch["input_ids"][0].numpy()
    labels = batch["labels"][0].numpy()

    # Check that input_ids and labels have the same content
    np.testing.assert_array_equal(input_ids, labels)


def test_prompt_masking(tokenizer):
    """Tests that prompt tokens are properly masked in labels when mask_prompt=True."""
    config = AlpacaDatasetConfig(
        data_source="yahma/alpaca-cleaned",
        max_examples=10,
        mask_prompt=True,
        seed=42
    )
    dataset = AlpacaDataset(config=config)
    dataset.setup(tokenizer)
    
    train_loader = dataset.train_dataloader()
    batch = next(iter(train_loader))

    # Get first sequence
    labels = batch["labels"][0].numpy()
    prompt_length = batch["prompt_length"][0].item()
    response_length = batch["response_length"][0].item()

    # Check prompt masking (excluding BOS token)
    assert (labels[1:prompt_length] == dataset.config.ignore_index).all(), \
        "Prompt section is not properly masked"

    # Check response section is not masked
    response_section = labels[prompt_length:prompt_length + response_length]
    assert not (response_section == dataset.config.ignore_index).all(), \
        "Response section should not be masked"


def test_validation_split(dataset):
    """Tests that train/validation split works correctly."""
    train_loader = dataset.train_dataloader()
    val_loader = dataset.val_dataloader()

    assert len(train_loader) > 0, "Train loader is empty"
    assert len(val_loader) > 0, "Validation loader is empty"
    assert len(train_loader) > len(
        val_loader
    ), "Train split should be larger than validation split"


def test_special_tokens_with_large_seq_length(tokenizer):
    """Tests that the special tokens are being added correctly."""
    config = AlpacaDatasetConfig(
        batch_size=1,
        max_seq_length=2048,
        data_source="yahma/alpaca-cleaned",
        max_examples=2,
        train_test_split=0.5,
    )
    dataset = AlpacaDataset(config=config)
    dataset.setup(tokenizer=tokenizer)

    train_loader = dataset.train_dataloader()
    batch = next(iter(train_loader))

    # Test for special tokens when truncation is unlikely
    sample_ids = batch["input_ids"][0]  # Get first sequence in batch
    assert (
        tokenizer.bos_token_id in sample_ids
    ), "BOS token not found in sequence"
    # assert (
    #     tokenizer.eos_token_id in sample_ids
    # ), "EOS token not found in sequence"

    # Verify token positions
    assert (
        sample_ids[0] == tokenizer.bos_token_id
    ), "BOS token should be at the start"
    # assert (
    #     sample_ids[-1] == tokenizer.eos_token_id
    # ), "EOS token should be at the end"
    # TODO(ntnsonti): EOS token is not getting added, take a look.
    # assert (tokenizer.eos_token_id
    #         in sample_ids), "EOS token not found in sequence"
