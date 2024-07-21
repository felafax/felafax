import torch
import torch_xla.core.xla_model as xm
from torch_xla.distributed.fsdp import XlaFullyShardedDataParallel as FSDP
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.parallel_loader as pl
from torch_xla.distributed.fsdp import (
    XlaFullyShardedDataParallel as FSDP,
    checkpoint_module,
)
from transformers import AutoModelForCausalLM, AutoConfig, GPT2LMHeadModel
from peft import LoraConfig, TaskType, get_peft_model

import os
from huggingface_hub import login

login(token="HUGGINGFACE_TOKEN")


def fsdp_wrapper(x):
    return FSDP(
        x,
        shard_param_on_dim_0=True,
        pin_layout_in_collective_ops=True,
        disable_reshard_on_root=False,
        reshard_after_forward=True,
    )


def apply_fsdp(module, class_names, fsdp_wrapper):
    for k, v in getattr(module, "_modules").items():
        if isinstance(v, (torch.nn.ModuleList, torch.nn.Sequential)):
            for i, layer in enumerate(v):
                clsname = layer.__class__.__name__
                if clsname in class_names:
                    v[i] = fsdp_wrapper(layer)
                    print("Applying FSDP on layer: ", layer.__class__.__name__)
                else:
                    apply_fsdp(layer, class_names, fsdp_wrapper)

        elif isinstance(v, torch.nn.Module):
            clsname = v.__class__.__name__

            if clsname in class_names:
                setattr(module, k, fsdp_wrapper(v))
                print("Applying FSDP on module: ", module.__class__.__name__)
            else:
                apply_fsdp(v, class_names, fsdp_wrapper)


def main(args):
    model_id = "TinyLlama/TinyLlama-1.1B-step-50K-105b"
    config = AutoConfig.from_pretrained(model_id, use_auth_token=True)
    module = AutoModelForCausalLM.from_config(config)

    # LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
    )
    module = get_peft_model(module, peft_config)
    module.print_trainable_parameters()

    apply_fsdp(module, ["LlamaDecoderLayer"], fsdp_wrapper)
    model = fsdp_wrapper(module)
    print(model)

    optim = torch.optim.Adam(model.parameters(), lr=0.0001)

    for input_ids_size in [256, 512, 1024]:
        print("Running forward pass...", input_ids_size)
        input_ids = torch.arange(input_ids_size).unsqueeze(0).to(xm.xla_device())

        output = model(input_ids=input_ids)
        loss = output.logits.mean()
        print(loss)

        loss.backward()
        optim.step()


if __name__ == "__main__":
    xmp.spawn(main)
    # main("")
