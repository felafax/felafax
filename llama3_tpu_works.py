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

login(token="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")

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-step-50K-105b"

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

def apply_fsdp(model):
    fsdp_util.apply_fsdp(model, ["LlamaDecoderLayer"])
    return model

def get_dataset(*, tokenizer, batch_size: int = 1):
    # Your existing get_dataset function here
    # (Commented out for brevity, but keep it in your actual code)
    pass

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

def fsdp_wrapper(x):
    return FSDP(x, shard_param_on_dim_0=True, pin_layout_in_collective_ops=True, disable_reshard_on_root=False, reshard_after_forward=True)

def train(index):
    torch.manual_seed(99)
    device = xm.xla_device()
    
    model, tokenizer = init_model(model_name=MODEL_NAME)
    model = apply_lora(model)
    model = apply_fsdp(model)
    model = fsdp_wrapper(model)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    
    # Uncomment these lines when you want to use your dataset
    # train_dataloader, test_dataloader = get_dataset(tokenizer=tokenizer, batch_size=1)
    # train_dataloader, test_dataloader = pl.MpDeviceLoader(train_dataloader, device), pl.MpDeviceLoader(test_dataloader, device)
    
    for epoch in range(1):
        xm.master_print(f'Epoch {epoch} train begin {test_utils.now()}')
        
        for i in [128, 256, 512, 1024, 2048]:
            optimizer.zero_grad()
            
            # Use this when you want to switch to your dataset
            # batch = next(iter(train_dataloader))
            # batch = {k: v.to(device) for k, v in batch.items()}
            
            input_ids = torch.arange(i).unsqueeze(0).to(device)
            
            output = model(input_ids=input_ids)
            loss = output.logits.mean()
            
            loss.backward()
            optimizer.step()
            
            xm.master_print(f'Loss: {loss:.2f}')

if __name__ == '__main__':
    xmp.spawn(train)