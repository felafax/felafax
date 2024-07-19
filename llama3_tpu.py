#!/usr/bin/env python
# coding: utf-8

# In[1]:


# get_ipython().system('pip install --upgrade pip -q')
# get_ipython().system('pip install transformers datasets sentencepiece peft -q')
# get_ipython().system('pip install huggingface_hub -q')
# get_ipython().system("pip uninstall tensorflow -y # If we don't do this, TF will take over TPU and cause permission error for PT")


# In[2]:


# get_ipython().system('export USE_TORCH=True # To use transformers library in TPU')


# In[3]:


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


# In[4]:


import sys
import importlib
sys.path.append('')
fsdp_util = importlib.import_module('utils.fsdp')
importlib.reload(fsdp_util)


# In[5]:


from huggingface_hub import login

login(token="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")


# In[6]:


MODEL_NAME = "TinyLlama/TinyLlama-1.1B-step-50K-105b" # "meta-llama/Llama-2-7b-hf"


# In[7]:


def init_model(*, model_name):
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
        config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer


# In[8]:


def apply_lora(model):
    # TODO: pass lora config as argument to function.
    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, 
                         inference_mode=False, 
                         r=8, 
                         lora_alpha=32, 
                         lora_dropout=0.1)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model


# In[9]:


def apply_fsdp(model):
    fsdp_util.apply_fsdp(model, ["LlamaDecoderLayer"])
    return model


# In[10]:


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


# In[23]:


def train():
    # dist.init_process_group('xla', init_method='xla://')
    torch.manual_seed(99)
    device = xm.xla_device()
    
    model, tokenizer = init_model(model_name="TinyLlama/TinyLlama-1.1B-step-50K-105b")
    model = apply_lora(model)
    model = apply_fsdp(model)

    train_dataloader, test_dataloader = get_dataset(tokenizer=tokenizer, batch_size=1)
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    train_dataloader, test_dataloader = pl.MpDeviceLoader(train_dataloader, device),  pl.MpDeviceLoader(test_dataloader, device)

    for epoch in range(1):
        xm.master_print('Epoch {} train begin {}'.format(epoch, test_utils.now()))
        
        for step, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            # TODO: set labels = batch['input_ids'][:, 1:] + -100
            output = model(input_ids=batch['input_ids'],
                           attention_mask=batch['attention_mask'],
                           labels=batch['input_ids'])
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            xm.master_print('Loss: {:.2f}%'.format(loss))


# In[24]:


def _map_fn(index):
    torch.set_default_dtype(torch.float32)
    train()


# In[ ]:


if __name__ == '__main__':
    xmp.spawn(_map_fn)


# In[ ]:


# model = model.cpu()
# print('now saving the model')
# model.push_to_hub(
#     "felarof01/llama3-test", 
#     tokenizer=tokenizer,
#     private=False,
#     create_pr=False,
#     max_shard_size="2GB", # Sharding isn't as important as before since hardware is better now but who cares anyway
# )

