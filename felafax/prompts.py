# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
import importlib
import re
from abc import abstractmethod
from json import dumps
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Tuple, Type, Union
from transformers import AutoTokenizer

import yaml


class PromptStyle:
    """Base interface for prompt styles."""

    @abstractmethod
    def apply(self, prompt: str, **kwargs: str) -> str:
        return prompt

    def stop_tokens(self, tokenizer: AutoTokenizer) -> Tuple[List[int], ...]:
        return ([tokenizer.eos_token_id], )

    @classmethod
    def from_name(cls, name: str) -> "PromptStyle":
        return prompt_styles[name]()


class Default(PromptStyle):

    def apply(self, prompt: str, **kwargs: str) -> str:
        return prompt

    def stop_tokens(self, tokenizer: AutoTokenizer) -> Tuple[List[int], ...]:
        return ([tokenizer.eos_token_id], )


class Alpaca(PromptStyle):

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
            f"### Instruction:\n{prompt}\n\n### Response:\n")


# Maps prompt style names to PromptStyle classes
prompt_styles: Dict[str, Type[PromptStyle]] = {
    # Dataset-specific prompt styles
    "alpaca": Alpaca,
}


def save_prompt_style(style: Union[str, PromptStyle],
                      checkpoint_dir: Path) -> None:
    style = PromptStyle.from_name(style) if isinstance(style, str) else style
    cls = type(style)
    # Allow saving the full module path for user-defined prompt classes
    config = {"class_path": f"{cls.__module__}.{cls.__name__}"}
    with open(checkpoint_dir / "prompt_style.yaml", "w",
              encoding="utf-8") as file:
        yaml.dump(config, file)


def load_prompt_style(checkpoint_dir: Path) -> PromptStyle:
    with open(checkpoint_dir / "prompt_style.yaml", "r",
              encoding="utf-8") as file:
        config = yaml.safe_load(file)
    # Support loading the full module path for user-defined prompt classes
    full_module_path, cls_name = config["class_path"].rsplit(".", 1)
    module = importlib.import_module(full_module_path)
    cls = getattr(module, cls_name)
    return cls()


def has_prompt_style(checkpoint_dir: Path) -> bool:
    return (checkpoint_dir / "prompt_style.yaml").is_file()
