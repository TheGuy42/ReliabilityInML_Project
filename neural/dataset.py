from __future__ import annotations
from torch.utils.data import Dataset, ConcatDataset
from torch import Tensor
import torch
import numpy as np
from lightning import LightningDataModule


import torch.utils.data as data
from lightning.pytorch.demos.boring_classes import RandomDataset


class DataModule(LightningDataModule):
    """
        A custom PyTorch Lightning DataModule class used to load and prepare given datasets for training, validation, and testing.

    """
    def __init__(
        self,
        dataset: Dataset,
        split: list[int],
        batch_size: int,
        num_workers: int = 0,
        shuffle: bool = True,
        seed: int = 42,
    ):
        super().__init__()
        self.dataset = dataset
        self.split = split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.seed = seed

    def prepare_data(self):
        # download, IO, etc. Useful with shared filesystems
        # only called on 1 GPU/TPU in distributed
        ...

    def setup(self, stage):
        # make assignments here (val/train/test split)
        # called on every process in DDP

        splits = data.random_split(
            self.dataset, self.split, generator=torch.Generator().manual_seed(self.seed)
        )
        if len(splits) == 2:
            self.train, self.val = splits
            self.test = None
        elif len(splits) == 3:
            self.train, self.val, self.test = splits
        else:
            raise ValueError("Split must be a list of length 2 or 3")

    def train_dataloader(self):
        return data.DataLoader(
            self.train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle,
        )

    def val_dataloader(self):
        return data.DataLoader(
            self.val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle,
        )

    def test_dataloader(self):
        return data.DataLoader(
            self.test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle,
        )

    def on_exception(self, exception):
        # clean up state after the trainer faced an exception
        ...

    def teardown(self):
        # clean up state after the trainer stops, delete files...
        # called on every process in DDP
        ...


class RandDataset(Dataset):
    """
    A custom Dataset class used to train the neural network. This class is used to create a PyTorch Dataset from a numpy array, and
    can be initialized with 'ndarrays' of the samples and labels, as well as a DatasetConfig configuration, in which the samples (X)
    and labels(y) will be created automatically.

    Args:
        X (np.ndarray): The input data as a numpy array.
        y (np.ndarray): The output data as a numpy array.
        config (DatasetConfig, optional): The configuration object for the dataset.
    """

    def __init__(self, data: Tensor, labels: Tensor):
        self.X = data
        self.y = labels

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx, :], self.y[idx, :]

    def save(self, path: str) -> None:
        torch.save(self, path)

    @staticmethod
    def load(path: str) -> None:
        return torch.load(path)

    @staticmethod
    def concatenate(datasets: list[RandDataset]) -> ConcatDataset:
        return ConcatDataset(datasets)

    @staticmethod
    def generate_normal_data(
        sampler: NormalSampler, shape: tuple[int, int], label: int
    ) -> RandDataset:
        data = sampler.sample(shape)
        labels = torch.ones(shape[0], 1) * label
        labels = labels.to(torch.int32)
        return RandDataset(data, labels)


class NormalSampler:
    def __init__(self, seed: int, mean: float, std: float):
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)
        self.set_mean(mean)
        self.set_std(std)

    def set_mean(self, mean: float) -> None:
        self.mean = mean

    def set_std(self, std: float) -> None:
        self.std = std

    def sample(self, shape: tuple) -> Tensor:
        return torch.randn(shape, generator=self.generator) * self.std + self.mean
