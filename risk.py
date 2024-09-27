from dataclasses import dataclass, field
import numpy as np
import torch
import torch.nn as nn
from typing import Callable, NewType
from confseq import predmix

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd



class Risk:
    """Risk
    This is a base class that represents a risk function.
    """
    def __init__(self):
        pass

    def __call__(self, features:np.ndarray, preds: np.ndarray, labels: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Risk class must implement a __call__ method")
    
    def __str__(self):
        return self.__class__.__name__


class MSE(Risk):
    def __init__(self):
        super().__init__()

    def __call__(self, features:np.ndarray, preds: np.ndarray, labels: np.ndarray) -> np.ndarray:
        return np.mean((preds - labels) ** 2)
    

class MAE(Risk):
    def __init__(self):
        super().__init__()

    def __call__(self, features:np.ndarray, preds: np.ndarray, labels: np.ndarray) -> np.ndarray:
        return np.mean(np.abs(preds - labels))
    

class Quantile(Risk):
    def __init__(self, q:float):
        super().__init__()
        self.q = q
    
    def __call__(self, features:np.ndarray, preds: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Compute the quantile loss.
        @param preds: The predicted values, shape (n_samples, 1).
        @param labels: The true values, shape (n_samples, 1).
        
        @return: The quantile loss.
        """
        res = np.maximum(self.q * (labels - preds), (self.q - 1) * (labels - preds))
        return np.mean(res)
    
    def __str__(self):
        return f"Quantile({self.q})"
    
    def __repr__(self):
        return f"Quantile({self.q})"


class RiskFilter(Risk):
    def __init__(self, risk:Risk, feature_cols:list[str], condition:Callable[[pd.DataFrame], pd.DataFrame], name:str=''):
        super().__init__()
        self.risk = risk
        self.condition = condition
        self.feature_cols = feature_cols
        self.name = name
    
    def __call__(self, features:np.ndarray, preds: np.ndarray, labels: np.ndarray) -> np.ndarray:
        features_df = pd.DataFrame(features, columns=self.feature_cols)
        mask = self.condition(features_df)
        filtered_features, filtered_preds, filtered_labels = features[mask], preds[mask], labels[mask]
        if len(filtered_preds) == 0:
            return np.nan
        return self.risk(filtered_features, filtered_preds, filtered_labels)

    def __str__(self):
        return f"RiskFilter:{self.name}"










