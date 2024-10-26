from typing import Any, Dict, Tuple, Optional
import torch
from datasets import load_dataset
from transformers import default_data_collator
from ...core.base_classes import BaseDatasetHandler, TrainingConfig
from ...core.distributed import BackendType
import jax.numpy as jnp


class LLaMADatasetHandler(BaseDatasetHandler):
    """Handler for LLaMA dataset processing."""

    PROMPT_TEMPLATE = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
    
    ### Instruction: {instruction}
    
    ### Input: {input}
    
    ### Response: {response}"""

    def __init__(self, backend: BackendType):
        self.backend = backend

    def get_dataset(
        self, data_source: str, tokenizer: Any, config: TrainingConfig
    ) -> Tuple[Any, Any]:
        """Get train and validation dataloaders."""
        # Load dataset
        if data_source.endswith(".json"):
            dataset = load_dataset("json", data_files=data_source)["train"]
        else:
            dataset = load_dataset(data_source)["train"]

        # Apply size limit if specified
        if config.dataset_size_limit:
            dataset = dataset.select(range(config.dataset_size_limit))

        # Format prompts
        dataset = dataset.map(
            lambda x: self._format_prompts(x, tokenizer),
            batched=True,
            remove_columns=dataset.column_names,
        )

        # Split dataset
        train_val_split = dataset.train_test_split(test_size=0.15)

        # Create dataloaders
        train_dataloader = self._create_dataloader(
            train_val_split["train"],
            tokenizer,
            config.batch_size,
            config.seq_length,
            shuffle=True,
        )

        val_dataloader = self._create_dataloader(
            train_val_split["test"],
            tokenizer,
            config.batch_size,
            config.seq_length,
            shuffle=False,
        )

        return train_dataloader, val_dataloader

    def _format_prompts(
        self, examples: Dict[str, list], tokenizer: Any
    ) -> Dict[str, list]:
        """Format examples using the prompt template."""
        texts = [
            self.PROMPT_TEMPLATE.format(
                instruction=instruction, input=input_text, response=response
            )
            + tokenizer.eos_token
            for instruction, input_text, response in zip(
                examples["instruction"], examples["input"], examples["output"]
            )
        ]

        return {"text": texts}

    def _create_dataloader(
        self,
        dataset: Any,
        tokenizer: Any,
        batch_size: int,
        seq_length: int,
        shuffle: bool = False,
    ) -> Any:
        """Create dataloader for the dataset."""
        # Tokenize dataset
        tokenized_dataset = dataset.map(
            lambda x: self._tokenize(x, tokenizer, seq_length),
            batched=True,
            remove_columns=dataset.column_names,
        )

        # Create appropriate dataloader based on backend
        if self.backend in [BackendType.JAX, BackendType.PYTORCH_XLA]:
            return torch.utils.data.DataLoader(
                tokenized_dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                collate_fn=self._collate_fn,
                drop_last=True,
            )
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    def _tokenize(
        self, examples: Dict[str, list], tokenizer: Any, seq_length: int
    ) -> Dict[str, list]:
        """Tokenize examples."""
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=seq_length + 1,  # +1 for shifted targets
        )

        return {
            "input_tokens": [ids[:-1] for ids in tokenized["input_ids"]],
            "target_tokens": [ids[1:] for ids in tokenized["input_ids"]],
            "attention_mask": [mask[:-1] for mask in tokenized["attention_mask"]],
        }

    def process_batch(self, batch: Dict[str, Any], tokenizer: Any) -> Dict[str, Any]:
        """Process a single batch of data."""
        if self.backend == BackendType.JAX:
            # Convert torch tensors to JAX arrays
            return {k: jnp.array(v.numpy()) for k, v in batch.items()}
        else:
            return batch

    def _collate_fn(self, examples: list) -> Dict[str, Any]:
        """Collate examples into a batch."""
        batch = default_data_collator(examples)

        # Convert to appropriate format based on backend
        if self.backend == BackendType.JAX:
            return {k: jnp.array(v.numpy()) for k, v in batch.items()}
        else:
            return batch
