import re

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as PS

###################################################
# Util functions for JAX RNG handling
###################################################
rng_generator = None


def init_rng(seed):
    global rng_generator
    rng_generator = NextRNG.from_seed(seed)


def next_rng(*args, **kwargs):
    global rng_generator
    return rng_generator(*args, **kwargs)


class NextRNG(object):
    """Stateful RNG generator, generate and delete within pure function."""

    @classmethod
    def from_seed(cls, seed):
        # Create new instance from a seed value
        return cls(jax.random.PRNGKey(seed))

    def __init__(self, rng):
        # Initialize with a JAX PRNG key
        self.rng = rng

    def __call__(self, keys=None):
        """Generates new RNG keys when the instance is called as a function."""
        if keys is None:
            # If no keys are provided, split the current RNG into two
            # Update the instance's RNG and return the new split RNG
            self.rng, split_rng = jax.random.split(self.rng)
            return split_rng

        elif isinstance(keys, int):
            # If an integer is provided, split the RNG into that many new keys plus one
            split_rngs = jax.random.split(self.rng, num=keys + 1)

            # Update the instance's RNG with the first split
            self.rng = split_rngs[0]

            # Return the remaining splits as a tuple
            return tuple(split_rngs[1:])
        else:
            # If a sequence of keys is provided, split the RNG into that many new keys plus one
            split_rngs = jax.random.split(self.rng, num=len(keys) + 1)

            # Update the instance's RNG with the first split
            self.rng = split_rngs[0]

            # Return a dictionary mapping the provided keys to the new RNG splits
            return {key: val for key, val in zip(keys, split_rngs[1:])}


###################################################
# Utils for JAX sharding
###################################################
# TODO: avoid defining mesh globally.
DEVICES = jax.devices()
DEVICE_COUNT = len(DEVICES)
DEVICE_MESH = mesh_utils.create_device_mesh(
    (DEVICE_COUNT // 2, DEVICE_COUNT // 2, 1))
MESH = Mesh(devices=DEVICE_MESH, axis_names=("dp", "fsdp", "mp"))


def apply_sharding_constraint(x, partition_spec):
    return jax.lax.with_sharding_constraint(
        x, NamedSharding(MESH, partition_spec))


def tree_path_to_string(path, sep=None):
    """Converts a JAX tree path to a string representation.
    
    Example: tree_path_to_string([DictKey('layer1'), SequenceKey(0)], sep='/') -> 'layer1/0'
    """
    keys = []
    for key in path:
        if isinstance(key, jax.tree_util.SequenceKey):
            keys.append(str(
                key.idx))  # Use index for sequences (lists, tuples)
        elif isinstance(key, jax.tree_util.DictKey):
            keys.append(str(key.key))  # Use actual key for dictionaries
        elif isinstance(key, jax.tree_util.GetAttrKey):
            keys.append(str(key.name))  # Use attribute name for objects
        elif isinstance(key, jax.tree_util.FlattenedIndexKey):
            keys.append(str(key.key))  # Use index for flattened arrays
        else:
            keys.append(str(key))  # Fallback: convert key to string directly

    if sep is None:
        return tuple(keys)  # Return as tuple if no separator
    return sep.join(keys)  # Join with separator if provided


def flatten_tree(xs, is_leaf=None, sep=None):
    """Flattens a JAX tree into a dictionary with path strings as keys."""
    flattened, _ = jax.tree_util.tree_flatten_with_path(xs, is_leaf=is_leaf)
    output = {}
    for key, val in flattened:
        output[tree_path_to_string(key, sep=sep)] = val
    return output


def named_tree_map(f, tree, is_leaf=None, sep='/'):
    """
    Maps a function over a JAX tree, providing both path and value to the function.
    
    Args:
        f: Function to apply to each node. It should accept (path, value) as arguments.
        tree: The tree structure to map over.
        is_leaf: Optional function to determine what constitutes a leaf in the tree.
        sep: Separator used in the string representation of the path.
    
    Returns:
        A new tree with f applied to each node.
    """

    # Helper function to process each node
    def process_node(path, value):
        # Convert the path to a string
        path_str = tree_path_to_string(path, sep=sep)
        return f(path_str, value)

    # Apply our helper function to the tree
    return jax.tree_util.tree_map_with_path(process_node,
                                            tree,
                                            is_leaf=is_leaf)


def match_partition_rules(rules, params):
    """Applies partitioning rules to a parameter tree."""

    def get_partition_spec(parm_path, param_value):
        # Don't partition scalar values
        if len(param_value.shape) == 0 or np.prod(param_value.shape) == 1:
            return NamedSharding(MESH, PS())

        for rule, ps in rules:
            if re.search(rule, parm_path) is not None:
                return NamedSharding(MESH, ps)

        raise ValueError(f'Partition rule not found for param: {parm_path}')

    # Apply get_partition_spec to each leaf in the parameter tree
    return named_tree_map(get_partition_spec, params, sep='/')


###################################################
# Loss and accuracy functions
###################################################


def cross_entropy_loss_and_accuracy(logits, tokens, valid=None):
    if valid is None:
        valid = jnp.ones(tokens.shape[:2])
    valid = valid.astype(jnp.float32)
    valid_text_length = jnp.maximum(jnp.sum(valid, axis=-1), 1e-10)
    logits = logits.astype(jnp.float32)  # for numerical stability
    token_log_prob = jnp.squeeze(
        jnp.take_along_axis(
            jax.nn.log_softmax(logits, axis=-1),
            jnp.expand_dims(tokens, -1),
            axis=-1,
        ),
        -1,
    )
    token_log_prob = jnp.where(valid > 0.0, token_log_prob, jnp.array(0.0))
    loss = -jnp.mean(jnp.sum(token_log_prob, axis=-1) / valid_text_length)
    correct = jnp.where(valid > 0.0,
                        jnp.argmax(logits, axis=-1) == tokens,
                        jnp.array(False))
    accuracy = jnp.mean(jnp.sum(correct, axis=-1) / valid_text_length)
    return loss, accuracy
