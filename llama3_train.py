import os
import contextlib
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch_xla.core.xla_model as xm
from torch_xla.distributed.fsdp import XlaFullyShardedDataParallel as FSDP, checkpoint_module
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.parallel_loader as pl
import torch_xla.test.test_utils as test_utils

from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer, default_data_collator
from datasets import Dataset, load_dataset, concatenate_datasets
from peft import LoraConfig, TaskType, get_peft_model

import logging
logging.getLogger("datasets").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)

import sys
import importlib
sys.path.append('')
fsdp_util = importlib.import_module('utils.fsdp')
importlib.reload(fsdp_util)

import os
from huggingface_hub import login

login(token="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")

supported_models = [
    "TinyLlama/TinyLlama-1.1B-step-50K-105b",
    "meta-llama/Llama-2-7b-hf",
    "meta-llama/Meta-Llama-3-8B",
]

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-step-50K-105b"


def init_model(*, model_name):
    config = AutoConfig.from_pretrained(model_name, use_auth_token=True)
    model = AutoModelForCausalLM.from_config(config)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
        config.pad_token_id = tokenizer.pad_token_id
    
    return model, tokenizer


def apply_lora(*, model, lora_rank=None, lora_alpha=None, lora_dropout=None):
    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM,
                             inference_mode=False,
                             r=8 if not lora_rank else lora_rank,
                             lora_alpha=32 if not lora_alpha else lora_alpha,
                             lora_dropout=0.1 if not lora_dropout else lora_dropout)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model

def apply_fsdp(*, model):
    # Apply on layers within model.
    fsdp_util.apply_fsdp(model, ["LlamaDecoderLayer"], fsdp_util.fsdp_wrapper)

    # Apply on the model itself.
    model = fsdp_util.fsdp_wrapper(model)
    return model

def get_dataset(*, tokenizer, batch_size=None, max_length=None):
    # Define Alpaca prompt template
    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
    
    ### Instruction: {}
    
    ### Input: {}
    
    ### Response: {}"""
    
    EOS_TOKEN = tokenizer.eos_token
    
    # Define formatting function.
    def _format_prompts(examples):
        instructions = examples["instruction"]
        inputs = examples["input"]
        outputs = examples["output"]
        texts = []
        for instruction, input, output in zip(instructions, inputs, outputs):
            text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
            texts.append(text)
        return {"text": texts}

    # Tokenize the dataset.
    def _tokenize(examples):
        tokenized = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512 if not max_length else max_length)
        # Compute labels for CausalLM.
        labels = tokenized["input_ids"].copy()
        for i in range(len(labels)):
            labels[i][:-1] = tokenized["input_ids"][i][1:]
            labels[i][-1] = -100
        tokenized["labels"] = labels
        return tokenized

    # Load and preprocess the dataset.
    dataset = load_dataset("yahma/alpaca-cleaned", split="train")
    dataset = dataset.map(_format_prompts, batched=True)

    # Create train and test dataset.
    ds = dataset.train_test_split(test_size=0.15)
    ds['train'] = ds['train'].map(_tokenize, batched=True, remove_columns=dataset.column_names)
    ds['test'] = ds['test'].map(_tokenize, batched=True, remove_columns=dataset.column_names)

    # Create DataLoader
    train_dataloader = torch.utils.data.DataLoader(
        ds['train'],
        shuffle=True,
        batch_size=1 if not batch_size else batch_size,
        collate_fn=default_data_collator,
    )
    
    test_dataloader = torch.utils.data.DataLoader(
        ds['test'],
        shuffle=True,
        batch_size=1 if not batch_size else batch_size,
        collate_fn=default_data_collator,
    )

    return train_dataloader, test_dataloader


def train(index):
    torch.manual_seed(99)
    device = xm.xla_device()
    
    trainer_config = {
        "lr": 5e-5,
        "batch_size": 1,
        "max_length": 512,
        
        "epochs": 1,
        
        "lora_rank": 8,
        "lora_alpha": 32,
        "lora_dropout": 0.1
    }
    
    model, tokenizer = init_model(model_name=MODEL_NAME)
    model = apply_lora(model=model, 
                       lora_rank=trainer_config["lora_rank"], lora_alpha=trainer_config["lora_alpha"], lora_dropout=trainer_config["lora_dropout"])
    model = apply_fsdp(model=model)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=trainer_config["lr"])

    train_dataloader, test_dataloader = get_dataset(tokenizer=tokenizer, batch_size=trainer_config["batch_size"], max_length=trainer_config["max_length"])
    train_dataloader, test_dataloader = pl.MpDeviceLoader(train_dataloader, device), pl.MpDeviceLoader(test_dataloader, device)

    for epoch in range(trainer_config["epochs"]):
        xm.master_print(f'Epoch {epoch} train begin {test_utils.now()}')
        
        model.train()
        for step, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            
            output = model(input_ids=batch['input_ids'],
                           attention_mask=batch['attention_mask'],
                           labels=batch['labels'])
            loss = output.loss
            
            loss.backward()
            optimizer.step()
            
            xm.master_print(f'Train Loss: {loss:.2f}')
        
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

if __name__ == "__main__":
    xmp.spawn(train)