import inspect
import logging
import os
import pprint
import random
import tempfile
import time
import uuid
from copy import copy, deepcopy
from io import BytesIO
from socket import gethostname

import absl.flags
import cloudpickle as pickle
import gcsfs
import numpy as np
from absl import logging
from ml_collections import ConfigDict
from ml_collections.config_dict.config_dict import placeholder as config_placeholder
from ml_collections.config_flags import config_flags


def config_dict(*args, **kwargs):
    return ConfigDict(dict(*args, **kwargs))


def define_flags_with_default(**kwargs):
    for key, val in kwargs.items():
        if isinstance(val, tuple):
            val, help_str = val
        else:
            help_str = ""

        if isinstance(val, ConfigDict):
            config_flags.DEFINE_config_dict(key, val)
        elif val == bool:
            absl.flags.DEFINE_bool(key, None, help_str)
        elif isinstance(val, bool):
            # Note that True and False are instances of int.
            absl.flags.DEFINE_bool(key, val, help_str)
        elif val == int:
            absl.flags.DEFINE_integer(key, None, help_str)
        elif isinstance(val, int):
            absl.flags.DEFINE_integer(key, val, help_str)
        elif val == float:
            absl.flags.DEFINE_float(key, None, help_str)
        elif isinstance(val, float):
            absl.flags.DEFINE_float(key, val, help_str)
        elif val == str:
            absl.flags.DEFINE_string(key, None, help_str)
        elif isinstance(val, str):
            absl.flags.DEFINE_string(key, val, help_str)
        else:
            raise ValueError("Incorrect value type")
    return absl.flags.FLAGS, kwargs


def get_user_flags(flags, flags_def):
    output = {}
    for key in flags_def:
        val = getattr(flags, key)
        if isinstance(val, ConfigDict):
            output.update(flatten_config_dict(val, prefix=key))
        else:
            output[key] = val

    return output

def update_config_dict(config, updates=None):
    updated_config = deepcopy(config)
    if updates is not None:
        updated_config.update(ConfigDict(updates).copy_and_resolve_references())
    return updated_config


def flatten_config_dict(config, prefix=None):
    output = {}
    for key, val in config.items():
        if isinstance(val, ConfigDict) or isinstance(val, dict):
            output.update(flatten_config_dict(val, prefix=key))
        else:
            if prefix is not None:
                output["{}.{}".format(prefix, key)] = val
            else:
                output[key] = val
    return output


def open_file(path, mode="rb", cache_type="readahead"):
    if path.startswith("gs://"):
        logging.getLogger("fsspec").setLevel(logging.WARNING)
        return gcsfs.GCSFileSystem().open(path, mode, cache_type=cache_type)
    else:
        return open(path, mode)


def makedirs(path, exist_ok=True):
    if path.startswith("gs://"):
        return gcsfs.GCSFileSystem().makedirs(path, exist_ok=exist_ok)
    else:
        return os.makedirs(path, exist_ok=exist_ok)


def save_pickle(obj, path):
    with open_file(path, "wb") as fout:
        pickle.dump(obj, fout)


def load_pickle(path):
    with open_file(path, "rb") as fin:
        data = pickle.load(fin)
    return data
