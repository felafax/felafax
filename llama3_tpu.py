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

import sys
import importlib
sys.path.append('')
fsdp_util = importlib.import_module('utils.fsdp')
importlib.reload(fsdp_util)

from huggingface_hub import login

login(token="HUGGINGFACE_TOKEN")

# "TinyLlama/TinyLlama-1.1B-step-50K-105b"
# "meta-llama/Meta-Llama-3-8B" 
MODEL_NAME = "meta-llama/Llama-2-7b-hf" 

def init_model(*, model_name):
    config = AutoConfig.from_pretrained(model_name, use_auth_token=True)
    model = AutoModelForCausalLM.from_config(config)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
        config.pad_token_id = tokenizer.pad_token_id
    
    return model, tokenizer

def apply_lora(model):
    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM,
                             inference_mode=False,
                             r=8,
                             lora_alpha=32,
                             lora_dropout=0.1)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model


def fsdp_wrapper(x):
    return FSDP(x, shard_param_on_dim_0=True, pin_layout_in_collective_ops=True, disable_reshard_on_root=False, reshard_after_forward=True)

def apply_fsdp(model):
    # Apply on layers within model.
    fsdp_util.apply_fsdp(model, ["LlamaDecoderLayer"])

    # Apply on the model itself.
    model = fsdp_wrapper(model)
    return model

def get_dataset(*, tokenizer, batch_size: int = 1):
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
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

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
        batch_size=batch_size,
        collate_fn=default_data_collator,
    )
    
    test_dataloader = torch.utils.data.DataLoader(
        ds['test'],
        shuffle=True,
        batch_size=batch_size,
        collate_fn=default_data_collator,
    )

    return train_dataloader, test_dataloader


def create_dummy_batch(batch_size=1, sequence_length=128):
    device = xm.xla_device()
    
    input_ids = torch.ones(batch_size, sequence_length, dtype=torch.long).to(device)
    attention_mask = torch.ones(batch_size, sequence_length, dtype=torch.long).to(device)
    labels = torch.ones(batch_size, sequence_length, dtype=torch.long).to(device)
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }



def train(index):
    torch.manual_seed(99)
    device = xm.xla_device()
    
    model, tokenizer = init_model(model_name=MODEL_NAME)
    model = apply_lora(model)
    model = apply_fsdp(model)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    
    train_dataloader, test_dataloader = get_dataset(tokenizer=tokenizer, batch_size=1)
    train_dataloader, test_dataloader = pl.MpDeviceLoader(train_dataloader, device), pl.MpDeviceLoader(test_dataloader, device)
    
    for epoch in range(1):
        xm.master_print(f'Epoch {epoch} train begin {test_utils.now()}')
        
        for step, batch in enumerate(train_dataloader):
            optimizer.zero_grad()

            # labels = batch['input_ids'].clone()
            # labels[:, :-1] = batch['input_ids'][:, 1:]
            # labels[:, -1] = -100
            # batch = {k: v.to(device) for k, v in batch.items()}
            
            output = model(input_ids=batch['input_ids'],
                           attention_mask=batch['attention_mask'],
                           labels=batch['input_ids'])
            loss = output.loss
            
            loss.backward()
            optimizer.step()
            
            xm.master_print(f'Loss: {loss:.2f}')

if __name__ == '__main__':
    xmp.spawn(train)
