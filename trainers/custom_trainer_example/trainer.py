from src.felafax.trainer_engine.trainer import Trainer


class CustomTrainer(Trainer):
    def configure_optimizers(self, optimizer_params):
        # Your custom optimizer configuration
        import optax

        # Custom optimizer configuration -- pass via TrainerConfig.
        warmup_steps = self.trainer_config.num_steps // 10
        total_steps = self.trainer_config.num_steps
        learning_rate = self.trainer_config.learning_rate
        weight_decay = self.trainer_config.weight_decay

        schedule_fn = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=learning_rate,
            warmup_steps=warmup_steps,
            decay_steps=total_steps,
        )

        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(learning_rate=schedule_fn, weight_decay=weight_decay),
        )
        self.optimizer = optimizer
        self.opt_state = self.optimizer.init(optimizer_params)
