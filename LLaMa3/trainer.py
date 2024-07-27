"""Mullti-controller trainer for LLaMa3 finetuning."""

from dataclasses import dataclass

import torch
import numpy as np
import torch.distributed as dist
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

xr.use_spmd()
assert xr.is_spmd() == True

import torch_xla.distributed.spmd as xs
from torch_xla.distributed.spmd import Mesh
from torch_xla.distributed.fsdp import checkpoint_module

import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.parallel_loader as pl
import torch_xla.test.test_utils as test_utils

from model import init_model, apply_lora
from dataset import get_dataset
import model_partitioning

# On a single TPU VM host, you can train/tune LLaMa 3/3.1 8B models with full precision or LoRA.
supported_models = [
    "TinyLlama/TinyLlama-1.1B-step-50K-105b",
    "meta-llama/Llama-2-7b-hf",
    "meta-llama/Meta-Llama-3-8B",
    "meta-llama/Meta-Llama-3.1-8B"
]

# Select a supported model from above list to use!
MODEL_NAME = "meta-llama/Meta-Llama-3-8B"
TRAINER_CONFIG = {
    "epochs": 1,
    "batch_size": 1,
    "max_length": 512,
    
    "lr": 5e-5,
    "logging_interval": 5,  # logs every 5 steps
    
    "lora_rank": 8,
    "lora_alpha": 32,
    "lora_dropout": 0.1,
}
HUGGINGFACE_TOKEN = "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"


def _print_training_update(device,
                          step,
                          loss,
                          rate,
                          global_rate,
                          epoch=None):
  """Prints the training metrics at a given step."""
  update_data = [
      'Training', 'Device={}'.format(_get_device_spec(device)),
      'Epoch={}'.format(epoch) if epoch is not None else None,
      'Step={}'.format(step), 'Loss={:.5f}'.format(loss),
      'Rate={:.2f}'.format(rate), 'GlobalRate={:.2f}'.format(global_rate),
      'Time={}'.format(now())
  ]
  print('|', ' '.join(item for item in update_data if item), flush=True)

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
    model_partitioning.partition_model(model=model, mesh=mesh)
    
    # Configure the training loop.
    optimizer = torch.optim.Adam(model.parameters(), lr=TRAINER_CONFIG["lr"])

    train_dataloader, test_dataloader = get_dataset(
        tokenizer=tokenizer,
        batch_size=TRAINER_CONFIG["batch_size"],
        max_length=TRAINER_CONFIG["max_length"],
    )
    train_dataloader, test_dataloader = pl.MpDeviceLoader(
        train_dataloader, 
        device,
    ) 
    test_dataloader = pl.MpDeviceLoader(
        test_dataloader, 
        device,
    )

    for epoch in range(TRAINER_CONFIG["epochs"]):
        xm.master_print(f"Epoch {epoch} train begin {test_utils.now()}")
        tracker = xm.RateTracker()
        
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

            if step%TRAINER_CONFIG["logging_interval"]==0:
                xm.add_step_closure(print_training_update, args=(device, step, loss, tracker, epoch))
            
        model.eval()
        eval_loss = 0
        with torch.no_grad():
            for step, batch in enumerate(test_dataloader):
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
                eval_loss += output.loss.item()
        avg_eval_loss = eval_loss / len(test_dataloader)
        xm.master_print(f'Epoch {epoch} eval loss: {avg_eval_loss:.2f}')
    return


if __name__ == "__main__":
    xmp.spawn(train)
