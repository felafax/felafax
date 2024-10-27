from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union, Iterable
from dataclasses import dataclass
from torch.utils.data import DataLoader, Dataset, IterableDataset


@dataclass
class DataConfig:
    """Configuration for data handling."""

    batch_size: int = 32
    num_workers: int = 0
    pin_memory: bool = True
    shuffle_train: bool = True


class BaseDatasetHandler(ABC):
    """Abstract base class for dataset handling, inspired by Lightning's DataModule.

    This class standardizes the training, validation, test splits, data preparation
    and transformations across different datasets.
    """

    def __init__(self, config: DataConfig = DataConfig()):
        self.config = config
        # These will be set in setup()
        self.train_dataset: Optional[Union[Dataset, Iterable[Dataset]]] = None
        self.val_dataset: Optional[Union[Dataset, Iterable[Dataset]]] = None
        self.test_dataset: Optional[Union[Dataset, Iterable[Dataset]]] = None
        self.predict_dataset: Optional[Union[Dataset, Iterable[Dataset]]] = None

    @abstractmethod
    def prepare_data(self) -> None:
        """Abstract method to handle data preparation.

        This method is called only once and on one GPU.
        Use this to download data, tokenize, etc.
        """
        pass

    @abstractmethod
    def setup(self, stage: Optional[str] = None) -> None:
        """Abstract method to handle dataset setup.

        This method is called on every GPU.
        Use this to make train/val/test/predict splits, initialize datasets, etc.

        Args:
            stage: Optional stage to setup. Can be 'fit', 'test', 'predict', or None
        """
        pass

    def train_dataloader(self) -> Optional[DataLoader]:
        """Creates the training dataloader."""
        if self.train_dataset is None:
            return None

        if isinstance(self.train_dataset, IterableDataset):
            shuffle = False
        else:
            shuffle = self.config.shuffle_train

        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
        )

    def test_dataloader(self) -> Optional[DataLoader]:
        """Creates the test dataloader."""
        if self.test_dataset is None:
            return None

        return DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
        )

    @abstractmethod
    def process_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Abstract method to process a batch of data.

        Args:
            batch: Dictionary containing batch data

        Returns:
            Processed batch dictionary
        """
        pass

    def state_dict(self) -> Dict[str, Any]:
        """Save datamodule state.

        Returns:
            Dictionary containing datamodule state
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load datamodule state.

        Args:
            state_dict: Dictionary containing datamodule state
        """
        pass

    def teardown(self, stage: Optional[str] = None) -> None:
        """Clean up after training/testing.

        Args:
            stage: Optional stage being torn down. Can be 'fit', 'test', 'predict', or None
        """
        pass

    def on_exception(self, exception: BaseException) -> None:
        """Handle any cleanup needed when an exception occurs.

        Args:
            exception: The exception that was raised
        """
        pass
