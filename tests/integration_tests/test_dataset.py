"""Tests for the dataset implementation in `base.py`."""

import pytest
import numpy as np
from transformers import AutoTokenizer

from src.felafax.trainer_engine.data.base import (
    DefaultDatasetLoader,
    DatasetConfig,
)

@pytest.fixture
def tokenizer():
    return AutoTokenizer.from_pretrained("felafax/tokenizer-llama-3.1-8B-Instruct-JAX")

@pytest.fixture
def dataset_config():
    """Provides dataset configuration for tests."""
    return DatasetConfig(
        data_source="yahma/alpaca-cleaned",
        batch_size=2,
        max_seq_length=2048,
        max_examples=10,
        prompt_style="alpaca",
        seed=42,
    )

@pytest.fixture
def dataset(tokenizer, dataset_config):
    return DefaultDatasetLoader(config=dataset_config, tokenizer=tokenizer)


def test_dataset_initialization(dataset):
    """Tests dataset initialization with basic config."""
    # Verify the dataset configuration matches expected values
    assert dataset.config.batch_size == 2
    assert dataset.config.max_seq_length == 2048

    # Check that train and validation datasets are initialized
    assert hasattr(dataset, "train_dataset")
    assert hasattr(dataset, "val_dataset")


def test_batch_keys_and_shapes(dataset):
    """Tests batch keys and tensor shapes."""
    train_loader = dataset.train_dataloader()
    batch = next(iter(train_loader))

    # Check required keys in batch
    assert "input_ids" in batch
    assert "labels" in batch

    # Verify batch dimensions match expected sizes
    assert batch["input_ids"].shape[0] == dataset.config.batch_size
    assert batch["labels"].shape[0] == dataset.config.batch_size
    assert batch["input_ids"].shape[1] <= dataset.config.max_seq_length
    assert batch["labels"].shape[1] <= dataset.config.max_seq_length


def test_labels_match_input_ids(dataset):
    """Tests that labels match input_ids where labels are not masked."""
    train_loader = dataset.train_dataloader()
    batch = next(iter(train_loader))

    # Get the first sequence from the batch
    input_ids = batch["input_ids"][0].numpy()
    labels = batch["labels"][0].numpy()

    # Identify valid positions (where labels are not masked)
    valid_positions = labels != dataset.config.ignore_index

    # Compare input_ids and labels at valid positions
    np.testing.assert_array_equal(
        input_ids[valid_positions],
        labels[valid_positions],
        err_msg="input_ids and labels do not match at valid positions",
    )


def test_prompt_tokens_are_masked(tokenizer):
    """Tests that prompt tokens are masked when mask_prompt=True."""
    alpaca_config = DatasetConfig(
        data_source="yahma/alpaca-cleaned",
        max_examples=10,
        mask_prompt=True,
        seed=42,
    )
    dataset = DefaultDatasetLoader(config=alpaca_config, tokenizer=tokenizer)

    train_loader = dataset.train_dataloader()
    batch = next(iter(train_loader))

    # Get labels and lengths from the batch
    labels = batch["labels"][0].numpy()
    prompt_length = batch["prompt_length"][0].item()
    response_length = batch["response_length"][0].item()

    # Check that prompt tokens are masked (exclude BOS token at position 0)
    assert (
        labels[1:prompt_length] == dataset.config.ignore_index
    ).all(), "Prompt section is not properly masked"

    # Check that response tokens are not masked
    response_section = labels[prompt_length : prompt_length + response_length]
    assert not (
        response_section == dataset.config.ignore_index
    ).all(), "Response section should not be masked"


def test_train_validation_split(dataset):
    """Tests that train/validation split works correctly."""
    train_loader = dataset.train_dataloader()
    val_loader = dataset.val_dataloader()

    # Verify that train and validation loaders have data
    assert len(train_loader) > 0, "Train loader is empty"
    assert len(val_loader) > 0, "Validation loader is empty"

    # Check that train split is larger than validation split
    assert len(train_loader) > len(
        val_loader
    ), "Train split should be larger than validation split"


def test_special_tokens_added(tokenizer):
    """Tests that special tokens are added correctly."""
    config = DatasetConfig(
        batch_size=1,
        max_seq_length=2048,
        data_source="yahma/alpaca-cleaned",
        max_examples=2,
        train_test_split=0.5,
    )
    dataset = DefaultDatasetLoader(config=config, tokenizer=tokenizer)

    train_loader = dataset.train_dataloader()
    batch = next(iter(train_loader))

    # Get the first sequence in the batch
    sample_ids = batch["input_ids"][0]

    # Check for presence of special tokens
    assert (
        tokenizer.bos_token_id in sample_ids
    ), "BOS token not found in sequence"

    eos_token_id = tokenizer.eos_token_id or tokenizer.pad_token_id
    assert eos_token_id in sample_ids, "EOS token not found in sequence"

    # Verify BOS token is at the start
    assert (
        sample_ids[0] == tokenizer.bos_token_id
    ), "BOS token should be at the start"


def test_sequences_padded_to_max_length(tokenizer):
    """Tests that sequences are padded to max_seq_length."""
    config = DatasetConfig(
        batch_size=2,
        max_seq_length=128,
        data_source="yahma/alpaca-cleaned",
        max_examples=4,
        mask_prompt=True,
        seed=42,
        pad_id=tokenizer.pad_token_id or 0,
    )
    dataset = DefaultDatasetLoader(config=config, tokenizer=tokenizer)

    train_loader = dataset.train_dataloader()
    batch = next(iter(train_loader))

    # Check that all sequences have length equal to max_seq_length
    assert (
        batch["input_ids"].shape[1] == config.max_seq_length
    ), "input_ids sequences are not of length max_seq_length"
    assert (
        batch["labels"].shape[1] == config.max_seq_length
    ), "labels sequences are not of length max_seq_length"

    # Iterate over each sample in the batch
    for i in range(config.batch_size):
        input_ids = batch["input_ids"][i]
        labels = batch["labels"][i]

        # Determine actual sequence length (excluding padding)
        input_seq_length = (input_ids != config.pad_id).sum().item()

        # Find positions where labels are not equal to ignore_index
        label_positions = (labels != config.ignore_index).nonzero(as_tuple=True)[
            0
        ]

        # Identify first and last positions of valid labels
        if label_positions.numel() > 0:
            first_label_pos = label_positions[0].item()
            last_label_pos = label_positions[-1].item()
        else:
            # If all labels are ignore_index
            first_label_pos = last_label_pos = -1

        # Verify non-padding elements are within max_seq_length
        assert (
            input_seq_length <= config.max_seq_length
        ), "input_ids sequence length exceeds max_seq_length"
        if last_label_pos != -1:
            assert (
                last_label_pos < config.max_seq_length
            ), "labels sequence length exceeds max_seq_length"

        # Verify input_ids are padded correctly after the sequence
        assert (
            input_ids[input_seq_length:] == config.pad_id
        ).all(), "input_ids padding values are incorrect"

        # Verify labels are ignore_index before first_label_pos and after last_label_pos
        assert (
            labels[:first_label_pos] == config.ignore_index
        ).all(), "labels initial padding values are incorrect"
        assert (
            labels[last_label_pos + 1 :] == config.ignore_index
        ).all(), "labels padding values are incorrect"
