import torch
import torch_xla.core.xla_model as xm
from torch_xla.distributed.fsdp import XlaFullyShardedDataParallel as FSDP
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.parallel_loader as pl

from transformers import AutoModelForCausalLM, AutoConfig, GPT2LMHeadModel
from fsdp_gpt import FSDPLlamaModel
from fsdp_peft import apply_fsdp
from peft import LoraConfig, TaskType, get_peft_model
from torch_xla.distributed.fsdp import XlaFullyShardedDataParallel as FSDP, checkpoint_module

import os
from dotenv import load_dotenv
from huggingface_hub import login

load_dotenv()
login(token=os.getenv("HUGGINGFACE_TOKEN"))

def fsdp_wrapper(x):
    return FSDP(x, shard_param_on_dim_0=True, pin_layout_in_collective_ops=True, disable_reshard_on_root=False, reshard_after_forward=True)

def main(args):
    model_id = "TinyLlama/TinyLlama-1.1B-step-50K-105b"
    config = AutoConfig.from_pretrained(model_id, use_auth_token=True)
    module = AutoModelForCausalLM.from_config(config)

    # LoRA!
    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
    module = get_peft_model(module, peft_config)
    module.print_trainable_parameters()

    apply_fsdp(module, ["LlamaDecoderLayer"], fsdp_wrapper)
    model = fsdp_wrapper(module)
    print(model)

    optim = torch.optim.Adam(model.parameters(), lr=0.0001)

    for i in [2048]: # [128, 256, 512, 1024, 2048]:
        print("forwarding...", i)
        input_ids = torch.arange(i).unsqueeze(0).to(xm.xla_device())

        output = model(input_ids=input_ids)

        # loss = output.last_hidden_state.mean()
        loss = output.logits.mean()
        print(loss)

        loss.backward()
        optim.step()

if __name__ == "__main__":
    xmp.spawn(main)
    # main("")