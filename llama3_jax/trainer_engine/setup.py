"""Sets up the environment for the training and provides utility functions."""

import importlib
import os
import sys


def setup_environment():
    os.environ["HF_HUB_CACHE"] = "/home/felafax-storage/hf/"
    os.environ["HF_HOME"] = "/home/felafax-storage/hf/"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Note: The following shell commands won't work directly in Python.
    # We'll use os.system to execute them.
    os.system('export HF_HUB_CACHE="/home/felafax-storage/hf/"')
    os.system('export HF_HOME="/home/felafax-storage/hf/"')
    os.system("export TOKENIZERS_PARALLELISM=false")


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


def reload_modules():
    clear_cache()
    import felafax
    import felafax.trainer_engine
    modules_to_reload = [
        module for name, module in sys.modules.items()
        if name.startswith('felafax')
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
