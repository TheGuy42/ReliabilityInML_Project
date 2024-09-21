from dataclasses import dataclass, field
import numpy as np
import torch
import torch.nn as nn
from typing import Callable, NewType
from confseq import predmix

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# from neural.module import LightningModel


class Risk:
    """Risk
    This is a base class that represents a risk function.
    """
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs) -> torch.Tensor | np.ndarray:
        raise NotImplementedError("Risk class must implement a __call__ method")




class MSE(Risk):
    def __init__(self):
        super().__init__()

    def __call__(self, preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return torch.mean((preds - labels) ** 2)
    

class MAE(Risk):
    def __init__(self):
        super().__init__()

    def __call__(self, preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.abs(preds - labels))
    

class Quantile(Risk):
    def __init__(self, q:float):
        super().__init__()
        self.q = q
    
    def __call__(self, preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute the quantile loss.
        @param preds: The predicted values, shape (n_samples, 1).
        @param labels: The true values, shape (n_samples, 1).
        
        @return: The quantile loss.
        """
        res = torch.max(self.q * (labels - preds), (self.q - 1) * (labels - preds))
        return torch.mean(res)















