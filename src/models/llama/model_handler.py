from typing import Any, Dict, Tuple, Optional
import jax
import jax.numpy as jnp
import torch
from transformers import AutoTokenizer
from ...core.base_classes import BaseModelHandler
from ...core.distributed import BackendType
from .config import LLaMAConfig, ModelRegistry


class LLaMAModelHandler(BaseModelHandler):
    """Handler for LLaMA model initialization and checkpointing."""

    def __init__(
        self,
        model_name: str,
        backend: BackendType,
        hf_token: Optional[str] = None,
        base_dir: str = "/tmp/llama",
    ):
        self.model_name = model_name
        self.backend = backend
        self.hf_token = hf_token
        self.base_dir = base_dir
        self.config = ModelRegistry.get_config(model_name)

    def load_model(self) -> Tuple[Any, Any]:
        """Load model and tokenizer based on backend type."""
        if self.backend == BackendType.JAX:
            return self._load_jax_model()
        elif self.backend == BackendType.PYTORCH_XLA:
            return self._load_pytorch_model()
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    def _load_jax_model(self) -> Tuple[Any, Any]:
        """Load JAX version of the model."""
        from .jax_impl.model import LLaMAForCausalLM

        # Initialize model with proper dtype
        model = LLaMAForCausalLM(
            config=self.config,
            dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16,
        )

        # Load tokenizer
        tokenizer = self._load_tokenizer()

        return model, tokenizer

    def _load_pytorch_model(self) -> Tuple[Any, Any]:
        """Load PyTorch version of the model."""
        from .pytorch_impl.model import LLaMAForCausalLM
        import torch_xla.core.xla_model as xm

        # Initialize model
        model = LLaMAForCausalLM(config=self.config)

        # Move model to XLA device
        device = xm.xla_device()
        model = model.to(device)

        # Load tokenizer
        tokenizer = self._load_tokenizer()

        return model, tokenizer

    def _load_tokenizer(self) -> Any:
        """Load tokenizer from HuggingFace."""
        hf_name = ModelRegistry.get_hf_name(self.model_name)
        tokenizer = AutoTokenizer.from_pretrained(
            hf_name, token=self.hf_token, cache_dir=self.base_dir
        )

        # Set padding token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return tokenizer

    def save_checkpoint(self, model_state: Any, path: str) -> None:
        """Save model checkpoint."""
        if self.backend == BackendType.JAX:
            self._save_jax_checkpoint(model_state, path)
        elif self.backend == BackendType.PYTORCH_XLA:
            self._save_pytorch_checkpoint(model_state, path)

    def _save_jax_checkpoint(self, model_state: Any, path: str) -> None:
        """Save JAX model checkpoint."""
        import flax

        # Only save on process 0
        if jax.process_index() == 0:
            checkpoint_dict = {
                "model": flax.serialization.to_bytes(model_state.params),
                "config": self.config.__dict__,
            }
            with open(path, "wb") as f:
                flax.serialization.msgpack_serialize_to_buffer(checkpoint_dict, f)

    def _save_pytorch_checkpoint(self, model_state: Any, path: str) -> None:
        """Save PyTorch model checkpoint."""
        import torch_xla.core.xla_model as xm

        # Only save on master process
        if xm.is_master_ordinal():
            checkpoint_dict = {
                "model": model_state.model.state_dict(),
                "optimizer": model_state.optimizer.state_dict(),
                "step": model_state.step,
                "config": self.config.__dict__,
            }
            xm.save(checkpoint_dict, path)

    def load_checkpoint(self, path: str) -> Any:
        """Load model checkpoint."""
        if self.backend == BackendType.JAX:
            return self._load_jax_checkpoint(path)
        elif self.backend == BackendType.PYTORCH_XLA:
            return self._load_pytorch_checkpoint(path)

    def _load_jax_checkpoint(self, path: str) -> Any:
        """Load JAX model checkpoint."""
        import flax

        with open(path, "rb") as f:
            checkpoint_dict = flax.serialization.msgpack_restore(f)

        params = flax.serialization.from_bytes(checkpoint_dict["model"])

        return params

    def _load_pytorch_checkpoint(self, path: str) -> Any:
        """Load PyTorch model checkpoint."""
        import torch_xla.core.xla_model as xm

        checkpoint_dict = torch.load(path, map_location="cpu")
        return checkpoint_dict
