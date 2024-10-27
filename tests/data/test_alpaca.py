import pytest
from transformers import AutoTokenizer

from felafax.trainer_engine.data.alpaca import AlpacaDataModule, AlpacaConfig
from felafax.prompts import PromptStyle


def test_alpaca_data_module():
    # Download the tokenizer from Hugging Face for llama-3.1-8B
    tokenizer = AutoTokenizer.from_pretrained(
        "felafax/tokenizer-llama-3.1-8B-Instruct-JAX")
    tokenizer.pad_token = tokenizer.eos_token  # Ensure pad_token is set

    # Configure the Alpaca Data Module
    config = AlpacaConfig(
        val_split_fraction=0.5,
        num_workers=2,
        batch_size=2,
        max_seq_length=10,
    )
    data_module = AlpacaDataModule(config=config)

    # Connect tokenizer and setup data
    data_module.connect(tokenizer=tokenizer)
    data_module.prepare_data()
    data_module.setup()

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
            0] == config.batch_size, f"Unexpected batch size for train_batch[{key}]"
        assert train_batch[key].shape[
            1] <= config.max_seq_length, f"Sequence length exceeds max_seq_length for {key}"

    # Test dataset attributes
    assert isinstance(train_dataloader.dataset.prompt_style,
                      PromptStyle), "Prompt style is not set correctly."

    # Optionally, test state management if implemented
    # state = data_module.state_dict()
    # new_data_module = AlpacaDataModule(config=config)
    # new_data_module.connect(tokenizer=tokenizer)
    # new_data_module.load_state_dict(state)
    # assert new_data_module.config == data_module.config
