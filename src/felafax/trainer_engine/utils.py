from huggingface_hub import create_repo, HfApi, HfFolder, upload_folder
import os
import jax

from typing import Optional


def upload_dir_to_hf(
    dir_path: str,
    repo_name: str,
    commit_message: str = "Add fine-tuned model",
    token: Optional[str] = None,
):
    """
    Uploads the model and tokenizer to Hugging Face Hub.

    Args:
        output_dir: Directory containing the model and tokenizer files.
        repo_name: Name of the repository on Hugging Face Hub (e.g., 'username/repo_name').
        commit_message: Commit message for the upload.
        token: Optional HF token for authentication.
    """
    api = HfApi()
    token = HfFolder.get_token() if token is None else token
    if token is None:
        raise ValueError(
            "You must either provide a token or be logged in to HuggingFace Hub to upload models."
        )

    # Create repo if it doesn't exist
    api.create_repo(repo_id=repo_name, exist_ok=True, token=token)

    # Upload the folder
    upload_folder(
        repo_id=repo_name,
        folder_path=dir_path,
        commit_message=commit_message,
        token=token,
        ignore_patterns=["*.py"],
    )

    print(
        f"Model uploaded to Hugging Face Hub at https://huggingface.co/{repo_name}"
    )


def named_tree_map(f, tree, is_leaf=None, sep="/"):
    """Maps a function over a JAX tree, providing path strings and values to the function.

    This function traverses a JAX tree structure and applies a function to each node,
    passing both the string representation of the node's path and its value.

    Args:
        f: Function taking (path_str, value) arguments to apply at each node
        tree: JAX tree structure to traverse
        is_leaf: Optional function to determine leaf nodes
        sep: Separator for path components (e.g., 'a/b/c' if sep='/')

    Example:
        >>> def print_node(path, val): print(f"{path}: {val}")
        >>> tree = {"layer1": {"w": 1, "b": 2}}
        >>> named_tree_map(print_node, tree)
        layer1/w: 1
        layer1/b: 2
    """

    def convert_path_key(key):
        """Converts a single path key to its string representation."""
        if isinstance(key, jax.tree_util.SequenceKey):
            return str(key.idx)
        elif isinstance(key, jax.tree_util.DictKey):
            return str(key.key)
        elif isinstance(key, jax.tree_util.GetAttrKey):
            return str(key.name)
        elif isinstance(key, jax.tree_util.FlattenedIndexKey):
            return str(key.key)
        return str(key)

    def process_node(path, value):
        """Processes a single node by converting its path and conditionally applying f."""
        path_str = sep.join(convert_path_key(k) for k in path)
        if is_leaf(value):
            return f(path_str, value)
        else:
            # TODO(lora): fix this
            return f(path_str, value)
            # return value

    return jax.tree_util.tree_map_with_path(process_node, tree, is_leaf=is_leaf)
