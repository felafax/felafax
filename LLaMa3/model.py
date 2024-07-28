"""Defines the model configuration for your LLaMa3 finetuning."""

from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model


def apply_lora(*, model, lora_rank=None, lora_alpha=None, lora_dropout=None):
    """Applies LoRA configuration to the model."""
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8 if not lora_rank else lora_rank,
        lora_alpha=32 if not lora_alpha else lora_alpha,
        lora_dropout=0.1 if not lora_dropout else lora_dropout,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model


def init_model(*, model_name, hugging_face_token):
    """Downloads and initializes the model."""
    config = AutoConfig.from_pretrained(
        model_name, 
        token=hugging_face_token)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        token=hugging_face_token,
        low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        token=hugging_face_token
    )

    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
        config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer
