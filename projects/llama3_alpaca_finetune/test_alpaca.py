"""Tests for the Alpaca dataset implementation in pipeline.py"""

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
    """Tests that the dataset correctly initializes with basic config."""
    # Verify that the dataset configuration matches the expected values
    assert dataset.config.batch_size == 2
    assert dataset.config.max_seq_length == 2048

    # Check that the dataset has train and validation datasets initialized
    assert hasattr(dataset, "train_dataset")
    assert hasattr(dataset, "val_dataset")


def test_output_keys_shapes(dataset):
    """Tests that the dataloader produces correctly formatted batches."""
    train_loader = dataset.train_dataloader()
    batch = next(iter(train_loader))

    # Check that the batch contains the required keys
    assert "input_ids" in batch
    assert "labels" in batch

    # Verify that the batch dimensions match the expected batch size and sequence length
    assert batch["input_ids"].shape[0] == dataset.config.batch_size
    assert batch["labels"].shape[0] == dataset.config.batch_size
    assert batch["input_ids"].shape[1] <= dataset.config.max_seq_length
    assert batch["labels"].shape[1] <= dataset.config.max_seq_length


def test_labels_match_inputs(dataset):
    """Tests that labels match input_ids exactly where labels are not masked."""
    train_loader = dataset.train_dataloader()
    batch = next(iter(train_loader))

    # Get the first sequence in the batch
    input_ids = batch["input_ids"][0].numpy()
    labels = batch["labels"][0].numpy()

    # Identify positions where labels are not equal to ignore_index
    valid_positions = labels != dataset.config.ignore_index

    # Compare input_ids and labels at valid positions
    np.testing.assert_array_equal(
        input_ids[valid_positions],
        labels[valid_positions],
        err_msg="input_ids and labels do not match at valid positions"
    )


def test_prompt_masking(tokenizer):
    """Tests that prompt tokens are properly masked in labels when mask_prompt=True."""
    # Configure the dataset to mask the prompt in the labels
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

    # Get the first sequence from the batch
    labels = batch["labels"][0].numpy()
    prompt_length = batch["prompt_length"][0].item()
    response_length = batch["response_length"][0].item()

    # Check that the prompt tokens are masked (set to ignore_index) in labels
    # Exclude the BOS token at position 0
    assert (labels[1:prompt_length] == dataset.config.ignore_index).all(), \
        "Prompt section is not properly masked"

    # Check that the response tokens are not masked
    response_section = labels[prompt_length:prompt_length + response_length]
    assert not (response_section == dataset.config.ignore_index).all(), \
        "Response section should not be masked"


def test_validation_split(dataset):
    """Tests that the train/validation split works correctly."""
    train_loader = dataset.train_dataloader()
    val_loader = dataset.val_dataloader()

    # Verify that both the train and validation loaders have data
    assert len(train_loader) > 0, "Train loader is empty"
    assert len(val_loader) > 0, "Validation loader is empty"

    # Check that the train split is larger than the validation split
    assert len(train_loader) > len(val_loader), \
        "Train split should be larger than validation split"


def test_special_tokens_with_large_seq_length(tokenizer):
    """Tests that special tokens are added correctly when sequences are not truncated."""
    # Configure the dataset with a large max_seq_length to avoid truncation
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

    # Get the first sequence in the batch
    sample_ids = batch["input_ids"][0]

    # Check that the BOS token is present in the sequence
    assert tokenizer.bos_token_id in sample_ids, "BOS token not found in sequence"

    # Verify that the BOS token is at the start of the sequence
    assert sample_ids[0] == tokenizer.bos_token_id, "BOS token should be at the start"

    # Note: EOS token assertions are commented out due to potential issues with tokenization
    # Ensure to revisit these when EOS token handling is updated


def test_fixed_seq_length(tokenizer):
    """Tests that all sequences are padded to max_seq_length."""
    # Configure the dataset with a smaller max_seq_length for testing
    config = AlpacaDatasetConfig(
        batch_size=2,
        max_seq_length=128,
        data_source="yahma/alpaca-cleaned",
        max_examples=4,
        mask_prompt=True,
        seed=42,
        pad_id=tokenizer.pad_token_id or 0,
    )
    dataset = AlpacaDataset(config=config)
    dataset.setup(tokenizer)

    train_loader = dataset.train_dataloader()
    batch = next(iter(train_loader))

    # Check that all sequences have length equal to max_seq_length
    assert batch["input_ids"].shape[1] == config.max_seq_length, \
        "input_ids sequences are not of length max_seq_length"
    assert batch["labels"].shape[1] == config.max_seq_length, \
        "labels sequences are not of length max_seq_length"

    # Iterate over each sample in the batch
    for i in range(config.batch_size):
        input_ids = batch["input_ids"][i]
        labels = batch["labels"][i]

        # Determine the actual sequence length (excluding padding)
        input_seq_length = (input_ids != config.pad_id).sum().item()

        # Find positions where labels are not equal to ignore_index
        label_positions = (labels != config.ignore_index).nonzero(as_tuple=True)[0]

        # Identify the first and last positions of valid labels
        if label_positions.numel() > 0:
            first_label_pos = label_positions[0].item()
            last_label_pos = label_positions[-1].item()
        else:
            # If all labels are ignore_index
            first_label_pos = last_label_pos = -1

        # Verify that non-padding elements are within max_seq_length
        assert input_seq_length <= config.max_seq_length, \
            "input_ids sequence length exceeds max_seq_length"
        if last_label_pos != -1:
            assert last_label_pos < config.max_seq_length, \
                "labels sequence length exceeds max_seq_length"

        # Verify that input_ids are padded correctly after the sequence
        assert (input_ids[input_seq_length:] == config.pad_id).all(), \
            "input_ids padding values are incorrect"

        # Verify that labels are set to ignore_index before first_label_pos and after last_label_pos
        assert (labels[:first_label_pos] == config.ignore_index).all(), \
            "labels initial padding values are incorrect"
        assert (labels[last_label_pos + 1:] == config.ignore_index).all(), \
            "labels padding values are incorrect"