import os
from ml_collections import ConfigDict
import cloudpickle as pickle
from copy import deepcopy

###################################################
# ConfigDict utils
###################################################


def create_config_dict(*args, **kwargs):
    """Creates and returns a ml_collections ConfigDict object.

    Example:
        # Using args
        config = create_config_dict({'learning_rate': 0.001, 'batch_size': 32})

        # Using kwargs
        config = create_config_dict(learning_rate=0.001, batch_size=32)

        # Using both args and kwargs
        base_config = {'model_type': 'transformer'}
        config = create_config_dict(base_config, num_layers=6, hidden_size=768)
    """
    return ConfigDict(dict(*args, **kwargs))


def update_config_dict(config: ConfigDict, updates: dict):
    """Creates an updated ConfigDict with applied changes."""
    updated_config = deepcopy(config)
    if updates is not None:
        updated_config.update(
            ConfigDict(updates).copy_and_resolve_references())
    return updated_config


def flatten_config_dict(config: ConfigDict, prefix=None):
    """Flattens a nested ConfigDict into a single-level dictionary."""
    output = {}
    for key, val in config.items():
        if isinstance(val, ConfigDict) or isinstance(val, dict):
            # Recursively flatten nested dictionaries
            output.update(flatten_config_dict(val, prefix=key))
        else:
            if prefix is not None:
                # Add prefix to key for nested values
                output["{}.{}".format(prefix, key)] = val
            else:
                # Add top-level key-value pairs directly
                output[key] = val
    return output


###################################################
# File system utils (supports GCS too)
###################################################


def open_file(path, mode="rb", cache_type="readahead"):
    if path.startswith("gs://"):
        raise NotImplementedError("GCS is not implemented yet.")
        import gcsfs
        return gcsfs.GCSFileSystem().open(path, mode, cache_type=cache_type)
    else:
        return open(path, mode)


def makedirs(path, exist_ok=True):
    if path.startswith("gs://"):
        raise NotImplementedError("GCS is not implemented yet.")
        import gcsfs
        return gcsfs.GCSFileSystem().makedirs(path, exist_ok=exist_ok)
    else:
        return os.makedirs(path, exist_ok=exist_ok)


###################################################
# Pickle utils
###################################################


def save_pickle(obj, path):
    """Saves an object to a pickle file at the specified path."""
    with open_file(path, "wb") as fout:
        pickle.dump(obj, fout)


def load_pickle(path):
    """Loads and returns an object from a pickle file at the specified path."""
    with open_file(path, "rb") as fin:
        data = pickle.load(fin)
    return data
