import torch
import torch_xla.core.xla_model as xm
from torch_xla.distributed.fsdp import XlaFullyShardedDataParallel as FSDP
from transformers import AutoModelForCausalLM, AutoConfig
from peft import LoraConfig, TaskType, get_peft_model


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
