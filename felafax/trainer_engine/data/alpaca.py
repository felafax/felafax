from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union
import json
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import requests
from tqdm import tqdm

from felafax.trainer_engine.data.base import BaseDatasetHandler, DataConfig


@dataclass
class AlpacaConfig(DataConfig):
    """Configuration for Alpaca dataset."""

    mask_prompt: bool = False
    val_split_fraction: float = 0.03865
    prompt_style: str = "alpaca"
    ignore_index: int = -100
    seed: int = 42
    download_dir: Path = Path("./data/alpaca")
    file_url: str = (
        "https://raw.githubusercontent.com/tloen/alpaca-lora/main/alpaca_data_cleaned_archive.json"
    )
    file_name: str = "alpaca_data_cleaned_archive.json"
    max_seq_length: int = -1


class AlpacaDataset(Dataset):
    """Dataset class for Alpaca data."""

    def __init__(
        self,
        data: list,
        tokenizer: Any,
        prompt_style: str,
        max_seq_length: int,
        mask_prompt: bool = False,
        ignore_index: int = -100,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.prompt_style = prompt_style
        self.max_seq_length = max_seq_length
        self.mask_prompt = mask_prompt
        self.ignore_index = ignore_index

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        # Here you would implement your specific tokenization and formatting
        # This is a placeholder for the actual implementation
        prompt = self._format_prompt(item)
        encoded = self._encode_text(prompt, item["output"])
        return encoded

    def _format_prompt(self, item: Dict[str, str]) -> str:
        # Implement prompt formatting based on prompt_style
        # This is a placeholder
        return f"Instruction: {item['instruction']}\nInput: {item['input']}\nOutput:"

    def _encode_text(self, prompt: str, response: str) -> Dict[str, torch.Tensor]:
        # Implement tokenization and encoding
        # This is a placeholder
        encoded = self.tokenizer(prompt + response, max_length=self.max_seq_length)
        return encoded


class AlpacaHandler(BaseDatasetHandler):
    """Handler for Alpaca dataset."""

    def __init__(self, config: Optional[AlpacaConfig] = None):
        self.config = config or AlpacaConfig()
        super().__init__(self.config)
        self.tokenizer = None

    def connect(self, tokenizer: Any) -> None:
        """Connect tokenizer to the handler."""
        self.tokenizer = tokenizer

    def prepare_data(self) -> None:
        """Download the dataset if needed."""
        self.config.download_dir.mkdir(parents=True, exist_ok=True)
        self._download_if_missing()

    def setup(self, stage: Optional[str] = None) -> None:
        """Set up train and validation datasets."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not connected. Call connect() first.")

        # Load data
        with open(
            self.config.download_dir / self.config.file_name, "r", encoding="utf-8"
        ) as file:
            data = json.load(file)

        # Split data
        train_data, val_data = random_split(
            data,
            [1.0 - self.config.val_split_fraction, self.config.val_split_fraction],
            generator=torch.Generator().manual_seed(self.config.seed),
        )

        # Create datasets
        self.train_dataset = AlpacaDataset(
            data=list(train_data),
            tokenizer=self.tokenizer,
            prompt_style=self.config.prompt_style,
            max_seq_length=self.config.max_seq_length,
            mask_prompt=self.config.mask_prompt,
            ignore_index=self.config.ignore_index,
        )

        self.val_dataset = AlpacaDataset(
            data=list(val_data),
            tokenizer=self.tokenizer,
            prompt_style=self.config.prompt_style,
            max_seq_length=self.config.max_seq_length,
            mask_prompt=self.config.mask_prompt,
            ignore_index=self.config.ignore_index,
        )

    def process_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Process a batch of data."""
        processed_batch = {
            key: value.to(self.device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        return processed_batch

    def _download_if_missing(self) -> None:
        """Downloads the dataset if not present."""
        file_path = self.config.download_dir / self.config.file_name
        if file_path.exists() and file_path.stat().st_size > 0:
            return

        response = requests.get(self.config.file_url, stream=True)
        with open(file_path, "w", encoding="utf-8") as f:
            pbar = tqdm(
                desc=str(file_path),
                total=int(response.headers.get("content-length", 0)),
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
            )
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data.decode())
                pbar.update(size)
            pbar.close()

    @property
    def device(self) -> torch.device:
        """Get the current device."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def state_dict(self) -> Dict[str, Any]:
        """Save handler state."""
        return {
            "config": self.config,
            "tokenizer_state": self.tokenizer.state_dict() if self.tokenizer else None,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load handler state."""
        self.config = state_dict["config"]
        if self.tokenizer and "tokenizer_state" in state_dict:
            self.tokenizer.load_state_dict(state_dict["tokenizer_state"])
