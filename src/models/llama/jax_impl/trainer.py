from typing import Any, Dict, Optional, Tuple
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from jax.experimental.pjit import with_sharding_constraint
from ....core.base_classes import BaseDistributedTrainer, TrainingConfig
from ....core.distributed import DistributedConfig
from ..config import LLaMAConfig

class LLaMAJAXTrainer(BaseDistributedTrainer):
    """JAX-based distributed trainer for LLaMA models."""
    
    def __init__(
        self,
        config: LLaMAConfig,
        training_config: TrainingConfig,
        dist_config: DistributedConfig,
    ):
        self.config = config
        self.training_config = training_config
        self.dist_config = dist_config
        self.mesh = dist_config.create_mesh()
        
    def setup_distributed(self) -> None:
        """Initialize distributed training environment."""
        jax.distributed.initialize()
    
    def create_state(
        self,
        model: Any,
        params: Dict[str, Any]
    ) -> train_state.TrainState:
        """Create training state."""
        # Create optimizer
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),  # Gradient clipping
            optax.adamw(
                learning_rate=self.training_config.learning_rate,
                b1=0.9,
                b2=0.999,
                eps=1e-8,
                weight_decay=0.01
            )
        )
        
        return train_state.TrainState.create(
            apply_fn=model.apply,
            params=params,
            tx=optimizer,
        )
    
    def _compute_loss(
        self,
        logits: jnp.ndarray,
        labels: jnp.ndarray,
        mask: jnp.ndarray
    ) -> Tuple[jnp.ndarray, Dict[str, float]]:
        """Compute loss and metrics."""
        # Convert logits to probabilities
        log_probs = jax.nn.log_softmax(logits)
        
        # Create one-hot encoded labels
        labels_onehot = jax.nn.one_hot(
            labels, num_classes=self.config.vocab_size
        )
        
        # Compute per-token loss
        loss = -jnp.sum(
            labels_onehot * log_probs,
            axis=-1
        )
        
        # Apply mask and normalize
        loss = jnp.sum(loss * mask) / jnp.sum(mask)
        
        # Compute accuracy
        predictions = jnp.argmax(logits, axis=-1)
        accuracy = jnp.sum(
            (predictions == labels) * mask
        ) / jnp.sum(mask)
        
        metrics = {
            "loss": loss,
            "accuracy": accuracy,
        }
        
        return loss, metrics
    
@partial(jax.jit, static_argnums=(0,))
    def train_step(
        self,
        state: train_state.TrainState,
        batch: Dict[str, jnp.ndarray],
        rng: Optional[Any] = None,
    ) -> Tuple[train_state.TrainState, Dict[str, float]]:
        """Execute single training step."""
        
        def loss_fn(params):
            logits = state.apply_fn(
                {"params": params},
                input_ids=batch["input_tokens"],
                attention_mask=batch["attention_mask"],
                deterministic=False,
            )
            
            loss, metrics = self._compute_loss(
                logits=logits,
                labels=batch["target_tokens"],
                mask=batch["attention_mask"]
            )
            return loss, metrics
        
        # Compute gradient
        (loss, metrics), grads = jax.value_and_grad(
            loss_fn, has_aux=True
        )(state.params)
        
        # Apply sharding constraint to gradients
        grads = with_sharding_constraint(
            grads,
            self.dist_config.sharding_config.partition_specs["gradients"]
        )
        
        # Update state
        state = state.apply_gradients(grads=grads)
        
        return state, metrics
    
    def train_epoch(
        self,
        state: train_state.TrainState,
        dataloader: Any,
        rng: Optional[Any] = None
    ) -> Tuple[train_state.TrainState, Dict[str, float]]:
        """Train for one epoch."""
        epoch_metrics = {
            "loss": 0.0,
            "accuracy": 0.0,
        }
        steps = 0
        
        for batch in dataloader:
            # Execute training step
            state, step_metrics = self.train_step(state, batch, rng)
            
            # Accumulate metrics
            for k, v in step_metrics.items():
                epoch_metrics[k] += v
            steps += 1
            
            # Log progress
            if steps % self.training_config.print_every_n_steps == 0:
                step_metrics = jax.device_get(step_metrics)
                print(f"Step {steps}: {step_metrics}")
            
            # Break if max steps reached
            if self.training_config.max_steps and steps >= self.training_config.max_steps:
                break
        
        # Average metrics
        for k in epoch_metrics:
            epoch_metrics[k] /= steps
            
        return state, epoch_metrics
    
    def evaluate(
        self,
        state: train_state.TrainState,
        dataloader: Any,
    ) -> Dict[str, float]:
        """Evaluate model."""
        eval_metrics = {
            "loss": 0.0,
            "accuracy": 0.0,
        }
        steps = 0
        
        for batch in dataloader:
            # Forward pass
            logits = state.apply_fn(
                {"params": state.params},
                input_ids=batch["input_tokens"],
                attention_mask=batch["attention_mask"],
                deterministic=True,
            )
            
            # Compute metrics
            _, step_metrics = self._compute_loss(
                logits=logits,
                labels=batch["target_tokens"],
                mask=batch["attention_mask"]
            )
            
            # Accumulate metrics
            for k, v in step_metrics.items():
                eval_metrics[k] += v
            steps += 1
            
            # Break if max eval steps reached
            if (self.training_config.max_eval_steps and 
                steps >= self.training_config.max_eval_steps):
                break
        
        # Average metrics
        for k in eval_metrics:
            eval_metrics[k] /= steps
            
        return eval_metrics
    
    def generate(
        self,
        state: train_state.TrainState,
        input_ids: jnp.ndarray,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> jnp.ndarray:
        """Generate text using the model."""
        
        @jax.jit
        def sample_token(logits: jnp.ndarray) -> jnp.ndarray:
            """Sample next token from logits."""
            # Apply temperature
            if temperature != 0:
                logits = logits / temperature
                
            # Apply top-k
            if top_k > 0:
                top_k_logits, top_k_indices = jax.lax.top_k(logits, top_k)
                logits = jnp.full_like(logits, float('-inf')).at[top_k_indices].set(top_k_logits)
            
            # Apply top-p (nucleus) sampling
            if 0 < top_p < 1.0:
                sorted_logits, sorted_indices = jax.lax.sort(logits, descending=True)
                cumulative_probs = jnp.cumsum(jax.nn.softmax(sorted_logits))
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove = jnp.roll(sorted_indices_to_remove, 1)
                sorted_indices_to_remove = sorted_indices_to_remove.at[0].set(False)
                indices_to_remove = jnp.zeros_like(logits, dtype=bool).at[sorted_indices].set(sorted_indices_to_remove)
                logits = logits.at[indices_to_remove].set(float('-inf'))
            
            # Sample from the modified distribution
            return jax.random.categorical(jax.random.PRNGKey(0), logits)
        
        # Initialize sequence with input_ids
        sequence = input_ids
        
        # Generate tokens auto-regressively
        for _ in range(max_length):
            # Get logits for the next token
            logits = state.apply_fn(
                {"params": state.params},
                input_ids=sequence,
                deterministic=True,
            )
            
            # Sample next token
            next_token = sample_token(logits[:, -1, :])
            
            # Append to sequence
            sequence = jnp.concatenate(
                [sequence, next_token[None, None]], 
                axis=1
            )
            
            # Stop if end token is generated
            if next_token == self.config.eos_token_id:
                break
        
       return sequence
