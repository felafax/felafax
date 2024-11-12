# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
import importlib
from abc import abstractmethod
from pathlib import Path
from typing import Dict, List, Tuple, Type, Union
from transformers import AutoTokenizer

import yaml


class BasePromptTemplate:
    """Base interface for prompt styles."""

    @abstractmethod
    def apply(self, prompt: str, **kwargs: str) -> str:
        return prompt

    def stop_tokens(self, tokenizer: AutoTokenizer) -> Tuple[List[int], ...]:
        return ([tokenizer.eos_token_id],)

    @classmethod
    def from_name(cls, name: str) -> "BasePromptTemplate":
        return prompt_templates[name]()


class DefaultPromptTemplate(BasePromptTemplate):
    def apply(self, prompt: str, **kwargs: str) -> str:
        return prompt

    def stop_tokens(self, tokenizer: AutoTokenizer) -> Tuple[List[int], ...]:
        return ([tokenizer.eos_token_id],)


class AlpacaPromptTemplate(BasePromptTemplate):
    def apply(self, prompt: str, **kwargs: str) -> str:
        if kwargs.get("input"):
            return (
                "Below is an instruction that describes a task, paired with an input that provides further context. "
                "Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{prompt}\n\n### Input:\n{kwargs['input']}\n\n### Response:\n"
            )
        return (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{prompt}\n\n### Response:\n"
        )


# Maps prompt style names to PromptStyle classes
prompt_templates: Dict[str, Type[BasePromptTemplate]] = {
    # Dataset-specific prompt styles
    "alpaca": AlpacaPromptTemplate,
}