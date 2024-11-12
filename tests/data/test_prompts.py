"""Tests for the prompt.py module.

pytest --capture=no tests/test_prompts.py
"""

import pytest
from transformers import AutoTokenizer
from felafax.trainer_engine.data.prompts import BasePromptTemplate, AlpacaPromptTemplate, DefaultPromptTemplate, prompt_templates


@pytest.fixture(scope="module")
def tokenizer():
    """Sets up the tokenizer for testing."""
    # Replace with the appropriate tokenizer for your models
    tokenizer = AutoTokenizer.from_pretrained(
        "felafax/tokenizer-llama-3.1-8B-Instruct-JAX"
    )
    return tokenizer


def test_default_prompt_style(tokenizer):
    """Tests that the Default PromptStyle works as expected with a real tokenizer."""
    prompt_style = DefaultPromptTemplate()
    prompt = "This is a test prompt."
    result = prompt_style.apply(prompt)
    assert (
        result == prompt
    ), "Default PromptStyle should return the prompt unchanged."

    stop_tokens = prompt_style.stop_tokens(tokenizer)
    assert isinstance(stop_tokens, tuple), "stop_tokens should return a tuple."
    assert (
        tokenizer.eos_token_id in stop_tokens[0]
    ), "EOS token should be in stop tokens."


def test_prompt_style_from_name():
    """Tests that prompt styles can be loaded from their registered names."""
    for style_name in prompt_templates:
        style_instance = BasePromptTemplate.from_name(style_name)
        assert isinstance(
            style_instance, prompt_templates[style_name]
        ), f"{style_name} did not load correctly."


def test_alpaca_prompt_format():
    """Tests that Alpaca prompt style formats messages correctly."""
    prompt = "Is a coconut a nut or a fruit?"
    style = AlpacaPromptTemplate()

    # Test without input
    output = style.apply(prompt)
    assert "Below is an instruction that describes a task." in output
    assert "### Instruction:" in output
    assert prompt in output
    assert "### Response:" in output

    # Test with input
    inp = "Consider its botanical classification."
    output = style.apply(prompt, input=inp)
    assert (
        "Below is an instruction that describes a task, paired with an input that provides further context."
        in output
    )
    assert "### Instruction:" in output
    assert "### Input:" in output
    assert prompt in output
    assert inp in output
    assert "### Response:" in output


class CustomPromptStyle(BasePromptTemplate):
    """Simple custom prompt style for testing."""

    def apply(self, prompt, **kwargs):
        return f"Custom: {prompt}"


def test_custom_prompt_style():
    """Tests that custom prompt styles can be created and used."""
    style = CustomPromptStyle()
    prompt = "Hello, world"
    result = style.apply(prompt)
    assert (
        result == "Custom: Hello, world"
    ), "CustomPromptStyle did not apply correctly."
