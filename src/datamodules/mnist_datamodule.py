"""MNIST DataModule using PyTorch's built-in dataset.

This module provides a LightningDataModule for the MNIST dataset,
suitable for quick testing and as a template for custom datasets.
"""

from typing import Optional

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms


class MNISTDataModule(LightningDataModule):
    """LightningDataModule for MNIST dataset.

    Args:
        data_dir: Directory to store/download the dataset
        batch_size: Batch size for train/val/test loaders
        num_workers: Number of workers for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
        persistent_workers: Whether to keep workers alive between epochs
    """

    def __init__(
        self,
        data_dir: str = "./data",
        batch_size: int = 64,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = False,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        # transforms
        self.train_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        self.test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

        self.data_train: Optional[torch.utils.data.Dataset] = None
        self.data_val: Optional[torch.utils.data.Dataset] = None
        self.data_test: Optional[torch.utils.data.Dataset] = None

    def prepare_data(self):
        """Download data if needed."""
        MNIST(self.hparams.data_dir, train=True, download=True)
        MNIST(self.hparams.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        """Load and split the dataset."""
        if stage == "fit" or stage is None:
            mnist_full = MNIST(
                self.hparams.data_dir,
                train=True,
                transform=self.train_transforms,
            )
            self.data_train, self.data_val = random_split(
                mnist_full,
                [55000, 5000],
                generator=torch.Generator().manual_seed(42),
            )

        if stage == "test" or stage is None:
            self.data_test = MNIST(
                self.hparams.data_dir,
                train=False,
                transform=self.test_transforms,
            )

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass