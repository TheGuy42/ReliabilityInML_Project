from __future__ import annotations
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader, random_split
from dataclasses import dataclass, field

from utils.config_base import ConfigBase
from utils.random_utils import Generators

@dataclass
class DatasetConfig(ConfigBase):
    featurs_dim: int 
    num_elements: int
    num_samples: int 

    @property
    def shape(self):
        return (self.num_samples, self.num_elements, self.featurs_dim)



OPTIMIZERS = {
    "adam": torch.optim.Adam,
    "sgd": torch.optim.SGD,
    "rmsprop": torch.optim.RMSprop,
    "adamw": torch.optim.AdamW,
}

LOSSES = {
    "mse": nn.MSELoss,
    "l1": nn.L1Loss,
    "cross_entropy": nn.CrossEntropyLoss,
    "bce": nn.BCELoss,
}


@dataclass
class ExperimentConfig(ConfigBase):
    # general parameters
    seed: int# = field(default=0, kw_only=True)  # Random seed for reproducibility
    data_seed: int# = field(default=42, kw_only=True)  # Random seed for reproducibility

    # model parameters 
    exp_name: str # The name of the experiment, used for logging and saving checkpoints
    # logs_dir: str = "logs/"  # The directory to save the logs and checkpoints


@dataclass
class TrainConfig(ConfigBase):
    # trainer parameters
    # loss_fn: str  # The loss function to use, can be any of the keys in the LOSSES dict
    # optimizer: str  # The optimizer to use, can be any of the keys in the OPTIMIZERS dict
    device: str = "cuda"  # 'cuda' for training on GPU or 'cpu' otherwise
    log: bool = False  # Whether to log and save the training process with tensorboard
    logs_dir: str = "logs/"  # The directory to save the logs and checkpoints

    # training parameters
    num_epochs: int = 100  # Number of times to iterate over the dataset
    checkpoints: bool = True  # Path to save model checkpoints, influding the checkpoint name.
    early_stopping: int = None  # Number of epochs to wait before stopping training if no improvement was made
    log_every: int = 5  # How often (#epochs) to print training progress
    timeout: str|None = None  # Maximum time (in seconds) to train the model, if None, it will train for num_epochs epochs

    # # optimizer parameters
    # learning_rate: float = 0.001  # Learning rate for the optimizer
    # weight_decay: float = 1e-5 # Weight decay for the optimizer (regularization, values typically in range [0.0, 1e-4] but can be bigger)

    # dataloader parameters
    batch_size: int = 256  # Number of samples in each batch
    shuffle: bool = True  # Whether to shuffle the dataset at the beginning of each epoch
    num_workers: int = 0  # Number of subprocesses to use for data loading
    # train_test_split: float = 0.8  # Fraction of the dataset to use for training, the rest will be used for testing




