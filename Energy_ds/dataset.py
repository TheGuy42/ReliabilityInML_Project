import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
from datetime import datetime
import os
from lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as data
import torch

from typing import Callable
from enum import Enum

from Energy_ds.config import REGION, SEASON, DatasetConfig



class DataPrep:
    def __init__(self, *file_paths:str):
        self.file_path = file_paths
        data = []
        for file in file_paths:
            data.append(self.prep(file))
        self.data = DataPrep.concat_data(*data)

    # Function to determine the season
    @staticmethod
    def get_season(date: datetime, north_hemisphere: bool = True) -> int:
        now = (date.month, date.day)
        if (3, 21) <= now < (6, 21):
            season = 'spring' if north_hemisphere else 'fall'
        elif (6, 21) <= now < (9, 21):
            season = 'summer' if north_hemisphere else 'winter'
        elif (9, 21) <= now < (12, 21):
            season = 'fall' if north_hemisphere else 'spring'
        else:
            season = 'winter' if north_hemisphere else 'summer'
        return SEASON.from_str(season).value

    @staticmethod
    def to_dataframe(file_path:str) -> pd.DataFrame:
        data = pd.read_csv(file_path)

        # Convert the 'Datetime' column to datetime object
        data['Datetime'] = pd.to_datetime(data['Datetime'])

        # Extract year, day, hour, and week number
        data['Year'] = data['Datetime'].dt.year
        data['Day'] = data['Datetime'].dt.day
        data['Hour'] = data['Datetime'].dt.hour
        data['Week_Number'] = data['Datetime'].dt.isocalendar().week

        # Apply the function to the 'Datetime' column
        data['Season'] = data['Datetime'].apply(DataPrep.get_season)

        # Display the first few rows to verify
        # print(data.head())
        return data

    @staticmethod
    def prep(file_path:str) -> pd.DataFrame:
        data = DataPrep.to_dataframe(file_path)
        data.sort_values(by='Datetime', inplace=True)
        data = data.reset_index(drop=True)
        region = os.path.basename(file_path).split('_')[0]
        data['Region'] = REGION.from_str(region).value
        data.rename(columns={f'{region}_MW':'MW'}, inplace=True)

        return data
    
    @staticmethod
    def concat_data(*data_frames: tuple[pd.DataFrame]) -> pd.DataFrame:
        data = pd.concat(data_frames, ignore_index=True)
        return data
    

class EnergyDataset(Dataset):
    feature_cols = ['Region', 'Year', 'Day', 'Hour', 'Week_Number', 'Season']
    target_col = 'MW'

    def __init__(self, 
                 data_path: str | list[str],
                 config: DatasetConfig = None,
                #  condition: pd.DataFrame | Callable[[pd.DataFrame], pd.DataFrame] = None,
                 transform: Callable[[pd.DataFrame], pd.DataFrame] = None,
                 ):
        self.data_path:list[str] = data_path if isinstance(data_path, list) else [data_path]
        self.config:DatasetConfig = config if config is not None else DatasetConfig()
        self.data = None
        self.labels = None

        self.data_df = None
        self.labels_df = None
        self.prepare_data(transform=transform)

    def prepare_data(self,
                    #  condition: pd.DataFrame | Callable[[pd.DataFrame], pd.DataFrame] = None,
                     transform: Callable[[pd.DataFrame], pd.DataFrame] = None,
                     ):
        prepper = DataPrep(*self.data_path)
        data = prepper.data

        features = data[self.feature_cols].copy()
        labels = data[self.target_col].copy()

        # Add past_hours labels as columns to the features dataframe
        for i in range(1, self.config.past_window + 1):
            col = f'MW_at_-{i}H'
            features[col] = labels.shift(i)

        # Drop rows with NaN values that were created by the shift operation
        features.dropna(inplace=True, ignore_index=True)
        labels = labels[self.config.past_window:].reset_index(drop=True)

        mask = self.config.conditions(features)
        # if condition is not None:
        #     if isinstance(condition, pd.DataFrame | pd.Series):
        #         mask = condition
        #     else:
        #         mask = condition(data)

        # apply the condition to a copy of the features and labels dataframes and save the results
        print(mask.shape)
        print(features.shape)
        print(labels.shape)
        self.data_df = features[mask].copy().reset_index(drop=True)
        self.labels_df = labels[mask].copy().reset_index(drop=True)
        
        # Convert the dataframes to tensors and apply the condition and transform functions
        self.data = self.to_tensor(features, condition=mask, transform=transform)
        self.labels = self.to_tensor(labels, condition=mask, transform=transform)

        # self.data, self.labels = self.transform_to_sequence(self.data, self.labels)

    def transform_to_sequence(self, data: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:    
        n_samples, n_features = data.shape
        if n_samples < self.past_hours:
            raise ValueError("Number of samples must be greater than or equal to past_hours")

        new_samples = n_samples - self.past_hours + 1
        transformed_data = np.zeros((new_samples, self.past_hours, n_features))
        transformed_labels = np.zeros((new_samples, 1))

        for i in range(new_samples):
            transformed_data[i] = data[i:i+self.past_hours]
            transformed_labels[i] = labels[i+self.past_hours-1]
        
        return transformed_data, transformed_labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    @staticmethod
    def to_tensor(data: pd.DataFrame,
                  condition: pd.DataFrame | Callable[[pd.DataFrame], pd.DataFrame] = None,
                  transform: Callable[[pd.DataFrame], pd.DataFrame] = None,
                  ) -> torch.Tensor:
        
        if transform is not None:
            data = transform(data)

        if condition is not None:
            if isinstance(condition, pd.DataFrame | pd.Series | np.ndarray):
                data = data[condition]
            else:
                data = data[condition(data)]

        data = data.to_numpy(dtype=np.float32, copy=True)
        return torch.from_numpy(data)




class DataModule(LightningDataModule):
    """
        A custom PyTorch Lightning DataModule class used to load and prepare given datasets for training, validation, and testing.

    """
    feature_cols = ['Region', 'Year', 'Day', 'Hour', 'Week_Number', 'Season']
    target_col = 'MW'

    def __init__(
        self,
        train_ds: Dataset,
        val_ds: Dataset,
        # data_path: str,
        # split: list[int],
        batch_size: int,
        num_workers: int = 0,
        shuffle: bool = True,
        seed: int = 42,
    ):
        super().__init__()
        self.train = train_ds
        self.val = val_ds
        # self.data_path = data_path
        # self.split = split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.seed = seed

    def prepare_data(self):
        # download, IO, etc. Useful with shared filesystems
        # only called on 1 GPU/TPU in distributed
       pass

    def setup(self, stage):
        # make assignments here (val/train/test split)
        # called on every process in DDP

        # splits = data.random_split(
        #     self.dataset, self.split, generator=torch.Generator().manual_seed(self.seed)
        # )
        # if len(splits) == 2:
        #     self.train, self.val = splits
        #     self.test = None
        # elif len(splits) == 3:
        #     self.train, self.val, self.test = splits
        # else:
        #     raise ValueError("Split must be a list of length 2 or 3")
        pass

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
        # return data.DataLoader(
        #     self.test,
        #     batch_size=self.batch_size,
        #     num_workers=self.num_workers,
        #     shuffle=self.shuffle,
        # )
        pass
















