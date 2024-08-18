# setup.py
import os

def setup_environment():
    os.environ['HF_HUB_CACHE'] = '/mnt/persistent-disk/hf/'
    os.environ['HF_HOME'] = '/mnt/persistent-disk/hf/'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    # Note: The following shell commands won't work directly in Python
    # We'll use os.system to execute them
    os.system('export HF_HUB_CACHE="/mnt/persistent-disk/hf/"')
    os.system('export HF_HOME="/mnt/persistent-disk/hf/"')
    os.system('export TOKENIZERS_PARALLELISM=false')
    

def setup_imports():
    # Standard library imports
    import os
    import pdb
    import enum
    import re
    import string
    from dataclasses import dataclass
    import functools
    from functools import partial
    from typing import (
        Any, List, Dict, Tuple, Optional, Union, Sequence, Mapping
    )

    # JAX and related libraries (including Flax and Optax)
    import jax
    import jax.numpy as jnp
    import flax
    import flax.linen as nn
    from flax.training.train_state import TrainState
    from flax.core.meta import unbox
    import optax
    import chex
    import lorax

    # JAX model partitioning and sharding
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as PS
    from jax.lax import with_sharding_constraint
    from jax.experimental import mesh_utils

    # Hugging Face Transformers and Datasets
    from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer, default_data_collator
    from datasets import Dataset, load_dataset, concatenate_datasets
    import torch

    # Gemma-specific imports
    from gemma import params as params_lib
    from gemma import sampler as sampler_lib
    from gemma import transformer as transformer_lib

    # Return a dictionary of all imported modules
    return locals()