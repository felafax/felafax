"""Defines the dataset pieline for your LLaMa3 finetuning."""

import torch
from transformers import (
    default_data_collator,
)
from datasets import load_dataset


def _format_prompts(examples, tokenizer):
    """Formats the prompts for the dataset."""
    # Define Alpaca prompt template
    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
    
    ### Instruction: {}
    
    ### Input: {}
    
    ### Response: {}"""
    EOS_TOKEN = tokenizer.eos_token

    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return {"text": texts}


def tokenize(examples, tokenizer, max_length):
    """Tokenizes the dataset."""
    # Tokenized is list within list. Compute labels for causalLM by shifting input_id; 
    # consequently truncate input_id to penultimate position.
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=512+1 if not max_length else max_length+1, # lenght+1 because last pos will be truncated in input_ids.
    )
    labels = tokenized['input_ids'].copy()
    tokenized['labels'] = [label[1:] for label in labels]
    tokenized['input_ids'] = [input_id[:-1] for input_id in tokenized['input_ids']]
    return tokenized


def get_dataset(*, tokenizer, batch_size=None, max_length=None):
    """Returns the training and test dataset loaders for LLaMa3 finetuning."""
    # Load and preprocess the dataset.
    dataset = load_dataset("yahma/alpaca-cleaned", split="train")
    dataset = dataset.map(lambda x: _format_prompts(x, tokenizer), batched=True)

    # Create train and test dataset.
    ds = dataset.train_test_split(test_size=0.15)
    ds["train"] = ds["train"].map(
        lambda x: tokenize(x, tokenizer, max_length),
        batched=True,
        remove_columns=dataset.column_names,
    )
    ds["test"] = ds["test"].map(
        lambda x: tokenize(x, tokenizer, max_length),
        batched=True,
        remove_columns=dataset.column_names,
    )

    # Create DataLoader
    train_dataloader = torch.utils.data.DataLoader(
        ds["train"],
        shuffle=True,
        batch_size=1 if not batch_size else batch_size,
        collate_fn=default_data_collator,
    )

    test_dataloader = torch.utils.data.DataLoader(
        ds["test"],
        shuffle=True,
        batch_size=1 if not batch_size else batch_size,
        collate_fn=default_data_collator,
    )

    return train_dataloader, test_dataloader
