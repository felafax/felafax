import os
import jax.numpy as jnp
import torch
from datasets import load_dataset
from transformers import default_data_collator
from typing import List, Dict, Any

class Dataset:
    ALPACA_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction: {}

    ### Input: {}

    ### Response: {}"""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.eos_token = tokenizer.eos_token

    def get_dataset(self, data_source, batch_size=1, seq_length=32, max_examples=None):
        # Load and preprocess the dataset
        if os.path.isfile(data_source):
            dataset = load_dataset("json", data_files=data_source, split="train")
        else:
            dataset = load_dataset(data_source, split="train")

        if max_examples:
            dataset = dataset.select(range(max_examples))
        
        dataset = dataset.map(self._format_prompts, batched=True)

        # Create train and test dataset
        ds = dataset.train_test_split(test_size=0.15)
        for split in ['train', 'test']:
            ds[split] = ds[split].map(
                lambda examples: self._tokenize(examples, seq_length),
                batched=True,
                remove_columns=dataset.column_names
            )

        # Create DataLoaders
        dataloader_args = dict(shuffle=True,
                               batch_size=batch_size,
                               collate_fn=self._custom_collate_fn)
        train_dataloader = torch.utils.data.DataLoader(ds['train'],
                                                       **dataloader_args)
        val_dataloader = torch.utils.data.DataLoader(ds['test'],
                                                     **dataloader_args)

        return train_dataloader, val_dataloader

    def _format_prompts(self, examples):
        instructions = examples["instruction"]
        inputs = examples["input"]
        outputs = examples["output"]
        texts = []
        for instruction, input, output in zip(instructions, inputs, outputs):
            text = self.ALPACA_PROMPT.format(instruction, input, output) + self.eos_token
            texts.append(text)
        return {"text": texts}

    def _tokenize(self, examples, seq_length):
        tokenized = self.tokenizer(examples["text"],
                                   truncation=True,
                                   padding="max_length",
                                   max_length=seq_length + 1)
        return {
            'input_tokens': [input_id[:-1] for input_id in tokenized['input_ids']],
            'target_tokens': [input_id[1:] for input_id in tokenized['input_ids']],
            'loss_masks': [input_id[1:] for input_id in tokenized['attention_mask']]
        }

    def _custom_collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, jnp.ndarray]:
        """
        Collates batch items and converts PyTorch tensors to JAX arrays.
        Applies default_data_collator, then converts tensors to JAX format.
        """
        collated = default_data_collator(batch)
        jax_batch = {}
        for key, value in collated.items():
            jax_batch[key] = jnp.array(value.numpy()) if isinstance(
                value, torch.Tensor) else value
        return jax_batch


def test_dataset_pipeline(tokenizer, data_source):
    """Print shapes of first batch to verify dataset pipeline."""
    dataset = Dataset(tokenizer)
    train_loader, _ = dataset.get_dataset(data_source=data_source,
                                          batch_size=4,
                                          seq_length=32,
                                          max_examples=32)
    batch = next(iter(train_loader))
    print("Input tokens shape:", batch['input_tokens'].shape)
    print("Target mask shape:", batch['target_tokens'].shape)
