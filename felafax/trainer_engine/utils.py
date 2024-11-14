from huggingface_hub import create_repo, HfApi, HfFolder, upload_folder
import os

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
    api.create_repo(repo_id=repo_name, exist_ok=True)

    # Upload the folder
    upload_folder(
        repo_id=repo_name,
        folder_path=dir_path,
        commit_message=commit_message,
        ignore_patterns=["*.py"],
    )

    print(
        f"Model uploaded to Hugging Face Hub at https://huggingface.co/{repo_name}"
    )
