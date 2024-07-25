"""Mullti-controller trainer for LLaMa3 finetuning."""

from dataclasses import dataclass

import torch
import numpy as np
import torch.distributed as dist
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

xr.use_spmd()
assert xr.is_spmd() == True

import torch_xla.experimental.xla_sharding as xs
from torch_xla.experimental.xla_sharding import Mesh
from torch_xla.distributed.fsdp import checkpoint_module

import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.parallel_loader as pl
import torch_xla.test.test_utils as test_utils

from model import init_model, apply_lora
from model_partitioning import apply_spmd
from dataset import get_dataset
from model_partitioning import apply_spmd, checkpoint_module


MODEL_NAME = "meta-llama/Meta-Llama-3.1-405B"
TRAINER_CONFIG = {
    "lr": 5e-5,
    "batch_size": 1,
    "max_length": 512,
    "epochs": 1,
    "lora_rank": 8,
    "lora_alpha": 32,
    "lora_dropout": 0.1,
}
HUGGINGFACE_TOKEN = "YOUR_HUGGINFACE_TOKEN"


def train(index):
    torch.manual_seed(99)
    device = xm.xla_device()

    # Create a mesh for the model partitioning.
    num_devices = xr.global_runtime_device_count()
    mesh_shape = (1, num_devices, 1)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("dp", "fsdp", "mp"))

    # Initialize the model and tokenizer.
    model, tokenizer = init_model(
        model_name=MODEL_NAME, hugging_face_token=HUGGINGFACE_TOKEN
    )
    model = checkpoint_module(model)
    
    # Apply LoRA configuration to the model.
    model = apply_lora(
        model=model,
        lora_rank=TRAINER_CONFIG["lora_rank"],
        lora_alpha=TRAINER_CONFIG["lora_alpha"],
        lora_dropout=TRAINER_CONFIG["lora_dropout"],
    )

    # Partition the model using SPMD.
    apply_spmd(model=model, mesh=mesh)
    
    # Configure the training loop.
    optimizer = torch.optim.Adam(model.parameters(), lr=TRAINER_CONFIG["lr"])

    train_dataloader, test_dataloader = get_dataset(
        tokenizer=tokenizer,
        batch_size=TRAINER_CONFIG["batch_size"],
        max_length=TRAINER_CONFIG["max_length"],
    )
    train_dataloader, test_dataloader = pl.MpDeviceLoader(
        train_dataloader, device
    ), pl.MpDeviceLoader(test_dataloader, device)

    for epoch in range(TRAINER_CONFIG["epochs"]):
        xm.master_print(f"Epoch {epoch} train begin {test_utils.now()}")

        model.train()
        for step, batch in enumerate(train_dataloader):
            optimizer.zero_grad()

            input_ids, attention_mask, labels = (
                batch["input_ids"],
                batch["attention_mask"],
                batch["labels"],
            )
            xs.mark_sharding(input_ids, mesh, (0, 1))
            xs.mark_sharding(attention_mask, mesh, (0, 1))
            xs.mark_sharding(labels, mesh, (0, 1))

            output = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            loss = output.loss

            loss.backward()
            optimizer.step()
            xm.mark_step()

            xm.master_print(f"Train Loss: {loss:.2f}")
            
            
            model.eval()
            eval_loss = 0
            with torch.no_grad():
                for step, batch in enumerate(test_dataloader):
                    output = model(input_ids=batch['input_ids'],
                                attention_mask=batch['attention_mask'],
                                labels=batch['labels'])
                    eval_loss += output.loss.item()
            
            avg_eval_loss = eval_loss / len(test_dataloader)
            xm.master_print(f'Epoch {epoch} eval loss: {avg_eval_loss:.2f}')
    return


if __name__ == "__main__":
    xmp.spawn(train)
