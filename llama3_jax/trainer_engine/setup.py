"""Sets up the environment for the training and provides utility functions."""

import importlib
import os
import sys


def setup_environment(base_dir):
    hf_dir = os.path.join(base_dir, "hf/")
    os.environ["HF_HUB_CACHE"] = hf_dir
    os.environ["HF_HOME"] = hf_dir
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Note: The following shell commands won't work directly in Python.
    # We'll use os.system to execute them.
    os.system(f"export HF_HUB_CACHE={hf_dir}")
    os.system(f"export HF_HOME={hf_dir}")
    os.system("export TOKENIZERS_PARALLELISM=false")

    jax_cache_dir = os.path.join(base_dir, "jax_cache/")
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_compilation_cache_dir", jax_cache_dir)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update("jax_threefry_partitionable", True)


def clear_cache():
    # Clear Python's import cache
    sys.modules.pop('felafax.trainer_engine', None)
    for key in list(sys.modules.keys()):
        if key.startswith('felafax.trainer_engine.'):
            sys.modules.pop(key, None)

    # Remove .pyc files
    current_dir = os.path.dirname(os.path.abspath(__file__))
    for root, dirs, files in os.walk(current_dir):
        for file in files:
            if file.endswith('.pyc'):
                os.remove(os.path.join(root, file))


def reload_modules(starts_with):
    clear_cache()
    modules_to_reload = [
        module for name, module in sys.modules.items()
        if name.startswith(starts_with)
    ]
    for module in modules_to_reload:
        try:
            importlib.reload(module)
        except ModuleNotFoundError as e:
            print(
                f"Warning: Could not reload module {module.__name__}. Error: {e}"
            )
    print("Reloaded all felafax modules.")


def main():
    setup_environment()
    reload_modules()


if __name__ == "__main__":
    main()
