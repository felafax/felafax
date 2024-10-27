from felafax.core.configs import (
    WorkflowConfig,
    TrainingConfig,
    ModelConfig,
    LoggingConfig,
    OptimizerConfig,
    DistributedConfig,
)

LEARNING_RATE = 1e-4
MODEL_NAME = "llama3-405b"


config = WorkflowConfig(
    name="llama3_full_finetune",
    model=ModelConfig(
        model_name=MODEL_NAME,
        use_lora=True,
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.1,
    ),
    training=TrainingConfig(
        learning_rate=LEARNING_RATE,
        num_epochs=3,
        batch_size=16,
        max_steps=1000,
        gradient_accumulation_steps=4,
        seq_length=2048,
        print_every_n_steps=10,
        eval_every_n_steps=100,
        max_eval_steps=50,
    ),
    distributed=DistributedConfig(
        backend="jax", dtype="bfloat16", param_dtype="bfloat16"
    ),
    optimizer=OptimizerConfig(
        name="adamw",
        weight_decay=0.01,
        max_grad_norm=1.0,
        warmup_steps=100,
        lr_scheduler="cosine",
    ),
    logging=LoggingConfig(
        wandb_project="llama-finetuning",
        log_every_n_steps=10,
        save_every_n_steps=100,
        eval_every_n_steps=500,
    ),
)

# Save to JSON
# config.to_json("config.json")
