"""Sets up the environment for the training and provides utility functions."""

import os
import jax
import warnings


def setup_environment(trainer_config):
    """
    Sets up the environment variables and JAX configurations.

    Args:
        trainer_config (TrainerConfig): Configuration object containing environment settings.
    """
    base_dir = trainer_config.base_dir

    # Setting up Hugging Face directories
    hf_dir = os.path.join(base_dir, "hf")
    os.makedirs(hf_dir, exist_ok=True)
    os.environ["HF_HUB_CACHE"] = hf_dir
    os.environ["HF_HOME"] = hf_dir
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Setting up JAX cache directories and configurations
    jax_cache_dir = os.path.join(base_dir, "jax_cache")
    os.makedirs(jax_cache_dir, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", jax_cache_dir)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update("jax_threefry_partitionable", True)

    # To suppress warning: RuntimeWarning: os.fork() was called. os.fork() is incompatible with multithreaded code, and JAX is multithreaded, so this will likely lead to a deadlock.
    warnings.filterwarnings("ignore", message="os.fork()")
