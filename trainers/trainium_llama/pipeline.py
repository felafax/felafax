# Standard library imports
from dataclasses import dataclass
from typing import Optional, Tuple

# Third-party imports
import torch
import torch_xla
import torch_xla.core.xla_model as xm
from transformers import (
    AutoModelForCausalLM,
    AutoConfig,
    AutoTokenizer,
    default_data_collator,
)
from peft import LoraConfig, TaskType, get_peft_model

# Local imports
from .dataset import create_dataloaders
from src.felafax.trainer_engine.data.data import DatasetConfig


@dataclass
class TrainerConfig:
    """Configuration for the Llama trainer"""

    # Model configuration
    model_name: str = "meta-llama/Llama-3.2-1B"
    param_dtype: str = "float32"
    compute_dtype: str = "float32"

    # Training configuration
    num_epochs: int = 1
    num_steps: Optional[int] = 10
    num_devices: int = 2
    mesh_shape: Optional[Tuple[int, int, int]] = None

    learning_rate: float = 1e-4

    # lora configuration
    lora_rank: int = 4  # Rank for lora matrices
    use_lora: bool = False  # Enable or disable lora training

    # Environment configuration
    base_dir: str = "/tmp/trainer_data/"
    hf_token: Optional[str] = None

    # Logging configuration
    log_interval: int = 1


def apply_lora(*, model, lora_rank=8, lora_alpha=32, lora_dropout=0.1):
    """Applies LoRA configuration to the model."""
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model


def init_model(*, model_name, hugging_face_token, trainer_config):
    """Downloads and initializes the model."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, token=hugging_face_token
    )

    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, token=hugging_face_token, low_cpu_mem_usage=True
    )

    # model = apply_lora(
    #     model=model,
    #     lora_rank=trainer_config.lora_rank,
    #     lora_alpha=trainer_config.lora_alpha,
    #     lora_dropout=trainer_config.lora_dropout,
    # )

    return model, tokenizer


def main():
    hf_token = input("Enter your Hugging Face token: ")

    trainer_config = TrainerConfig(
        hf_token=hf_token,
    )

    model, tokenizer = init_model(
        model_name=trainer_config.model_name,
        hugging_face_token=trainer_config.hf_token,
    )

    # Create dataset configuration for MedQA
    medqa_config = DatasetConfig(
        # Data loading parameters
        data_source="ngram/medchat-qa",
        max_examples=None,
        # Batching parameters
        batch_size=8,
        max_seq_length=32,
        num_workers=8,
        ignore_index=-100,
        mask_prompt=False,
        pad_id=0,
    )
    train_dataloader, val_dataloader = create_dataloaders(
        config=medqa_config, tokenizer=tokenizer
    )

    torch.manual_seed(99)
    device = xm.xla_device()
    model = model.to(device)

    optimizer = torch.optim.SGD(
        model.parameters(), lr=trainer_config.learning_rate
    )

    max_steps = trainer_config.num_steps or float("inf")
    step = 0
    prev_step = -1
    prev_loss = 0.0

    for epoch in range(trainer_config.num_epochs):
        model.train()

        for step, batch in enumerate(train_dataloader):
            if step > max_steps:
                break
            if (prev_step + 1) % trainer_config.log_interval == 0:
                xm.master_print(f"Step {prev_step} loss: {prev_loss}")

            optimizer.zero_grad()
            input_ids, attention_mask, labels = (
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device)
                if "attention_mask" in batch
                else None,
                batch["labels"].to(device),
            )

            output = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            loss = output.loss
            loss.backward()
            optimizer.step()
            xm.mark_step()

            prev_step = step
            prev_loss = loss.detach().cpu().item()
            step = step + 1

    print(f"Training complete! Final loss: {loss.detach().to('cpu')}")


if __name__ == "__main__":
    main()
