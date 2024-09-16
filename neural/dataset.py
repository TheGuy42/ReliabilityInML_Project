from __future__ import annotations
from torch.utils.data import Dataset, ConcatDataset
from torch import Tensor
import torch
import numpy as np
from lightning import LightningDataModule

from neural.symmetry import Sn, SYM_TYPE

    

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
    def generate_normal_data(sampler: NormalSampler,shape: tuple[int, int], label: int) -> RandDataset:
        data = sampler.sample(shape)
        labels = torch.ones(shape[0], 1) * label
        labels = labels.to(torch.int32)
        return RandDataset(data, labels)
    
    def augment(self, n:int) -> RandDataset:
        datasets = []
        sn = Sn(self.X.shape[1], n_perms=n, sample=True, sym_type=SYM_TYPE.INVARIANT)
        data = self.X.clone()
        labels = self.y.clone()
        for _ in sn:
            aug_data = sn(data)
            datasets.append(RandDataset(aug_data, labels))
        
        return RandDataset.concatenate(datasets)

        
        

    
class NormalSampler:
    def __init__(self, seed:int, mean:float, std:float):
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)
        self.set_mean(mean)
        self.set_std(std)

    def set_mean(self, mean:float) -> None:
        self.mean = mean
    
    def set_std(self, std:float) -> None:
        self.std = std

    def sample(self, shape:tuple) -> Tensor:
        return torch.randn(shape, generator=self.generator)*self.std + self.mean





