import pytest
import torch
from pathlib import Path
from typing import Any, Dict

from felafax.trainer_engine.data.alpaca import AlpacaHandler, AlpacaConfig

# from prompt_styles import AlpacaPromptStyle  # You'll need to implement this


class MockTokenizer:
    """Mock tokenizer for testing."""

    def __init__(self):
        self.vocab_size = 100

    def __call__(self, text: str, max_length: int) -> Dict[str, torch.Tensor]:
        # Mock tokenization - returns fixed tensors
        batch_size = 1
        return {
            "input_ids": torch.randint(0, self.vocab_size, (batch_size, max_length)),
            "labels": torch.randint(0, self.vocab_size, (batch_size, max_length)),
            "token_counts": torch.tensor([max_length]),
        }

    def state_dict(self) -> Dict[str, Any]:
        return {"vocab_size": self.vocab_size}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.vocab_size = state_dict["vocab_size"]


@pytest.fixture
def mock_tokenizer():
    return MockTokenizer()


@pytest.fixture
def alpaca_path(tmp_path):
    """Create a temporary path and mock dataset file."""
    data_path = tmp_path / "alpaca_data.json"
    mock_data = [
        {
            "instruction": "Test instruction 1",
            "input": "Test input 1",
            "output": "Test output 1",
        }
    ] * 24  # Creating 24 identical samples to match the test requirements

    import json

    with open(data_path, "w", encoding="utf-8") as f:
        json.dump(mock_data, f)

    return data_path


def test_alpaca(mock_tokenizer, alpaca_path):
    """Test the AlpacaHandler implementation."""
    # Initialize handler with config
    config = AlpacaConfig(
        val_split_fraction=0.5,
        download_dir=alpaca_path.parent,
        file_name=alpaca_path.name,
        num_workers=0,
        batch_size=2,
        max_seq_length=10,
    )
    handler = AlpacaHandler(config)

    # Test prompt style
    # assert isinstance(handler.config.prompt_style, str)
    # assert handler.config.prompt_style == "alpaca"

    # Connect tokenizer and setup data
    handler.connect(mock_tokenizer)
    handler.prepare_data()
    handler.setup()

    # Get dataloaders
    train_dataloader = handler.train_dataloader()

    # Test dataloader lengths
    assert len(train_dataloader) == 6  # 12 samples / batch_size of 2

    # Test batch structure
    train_batch = next(iter(train_dataloader))

    # Test batch keys
    expected_keys = {"input_ids", "labels", "token_counts"}
    assert (
        train_batch.keys() == expected_keys
    ), f"Train batch has unexpected keys: {train_batch.keys()}"

    # Test batch shapes
    # for key in ["input_ids", "labels"]:
    #     assert train_batch[key].shape == (
    #         2,
    #         10,
    #     ), f"Unexpected shape for train_batch[{key}]"

    # Test dataset attributes
    assert isinstance(train_dataloader.dataset.prompt_style, str)

    # Test state management
    state = handler.state_dict()
    new_handler = AlpacaHandler(config)
    new_handler.connect(MockTokenizer())
    new_handler.load_state_dict(state)
    assert new_handler.config == handler.config


def test_alpaca_handler_errors():
    """Test error handling in AlpacaHandler."""
    handler = AlpacaHandler()

    # Test missing tokenizer error
    with pytest.raises(ValueError, match="Tokenizer not connected"):
        handler.setup()

    # Test invalid file path
    config = AlpacaConfig(
        download_dir=Path("/nonexistent"), file_name="nonexistent.json"
    )
    handler = AlpacaHandler(config)
    with pytest.raises(
        Exception
    ):  # Specific exception type depends on your implementation
        handler.prepare_data()


def test_alpaca_handler_batch_processing(mock_tokenizer, alpaca_path):
    """Test batch processing functionality."""
    config = AlpacaConfig(
        download_dir=alpaca_path.parent,
        file_name=alpaca_path.name,
        batch_size=2,
        max_seq_length=10,
    )
    handler = AlpacaHandler(config)
    handler.connect(mock_tokenizer)
    handler.prepare_data()
    handler.setup()

    # Get a batch
    train_loader = handler.train_dataloader()
    batch = next(iter(train_loader))

    # Process batch
    processed_batch = handler.process_batch(batch)

    # Check processed batch
    assert isinstance(processed_batch, dict)
    assert all(torch.is_tensor(v) for v in processed_batch.values())
    assert all(
        v.device == handler.device
        for v in processed_batch.values()
        if torch.is_tensor(v)
    )
