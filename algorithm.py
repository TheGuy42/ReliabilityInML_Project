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

    @property
    def value(self):
        pass

class ConfSeq:
    """
    This is a base class that represents a bound function.
    """
    def __init__(self, confidence:float, bound_name:str=None):
        self.name = bound_name if bound_name is not None else self.__class__.__name__
        self.conf_lvl = confidence

        self._risk_seq:np.ndarray = np.ndarray([]).reshape(1)
        self._lower_cs:np.ndarray = np.ndarray([]).reshape(1)
        self._upper_cs:np.ndarray = np.ndarray([]).reshape(1)

    def calculate_cs(self, x:np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate the confidence interval bounds given a risk sequence x.
        """
        raise NotImplementedError("Bound class must implement _update_bounds method")

    def update(self, x:np.ndarray|float) -> tuple[np.ndarray, np.ndarray]:
        """
        update the confidence interval bounds given a new input x.
        """
        self._risk_seq = self._append_seq(x, self._risk_seq)

        self._lower_cs, self._upper_cs = self.calculate_cs(self._risk_seq)

        return self.lower, self.upper
    
    def _append_seq(self, x:np.ndarray|float, seq:np.ndarray) -> np.ndarray:
        """
        append the input x to the sequence.
        parameters:
            x (np.ndarray): The input array of shape (n_samples) or a float if x is a single sample.
            seq (np.ndarray): The sequence to append to.
        """
        if isinstance(x, float):
            x = np.ndarray([x])
        # print(seq.shape, x.shape)
        new_seq = np.concatenate((seq, x), axis=0)
        # new_seq = np.stack((seq.copy(), x), axis=0)

        return new_seq

    @property
    def lower(self):
        return self._lower_cs

    @property
    def upper(self):
        return self._upper_cs
    
    def to_dataframe(self):
        df = pd.DataFrame(columns=["data", "lower_ci", "upper_ci"])
        df['time'] = np.arange(self._risk_seq.shape[0])
        df["data"] = self._risk_seq
        df["lower_ci"] = self._lower_cs
        df["upper_ci"] = self._upper_cs
        return df
    
    def plot(self):
        fig, axes = plt.subplots(1, 1, figsize=(12, 6))
        fig.set_dpi(100)

        df = self.to_dataframe()

        # g = sns.relplot(data=df, x='time', y=df.columns, hue=df.columns, ax=axes)
        g = sns.lineplot(data=df, x='time', y='data', ax=axes, label='Risk Sequence')
        g.fill_between(df['time'], df['lower_ci'], df['upper_ci'], alpha=0.35, color='red', label='Confidence Interval', zorder=10)
        g.set_title(f"{self.name} Confidence Interval; Confidence Level: {self.conf_lvl}; Coverage: {self.coverage()}")
        g.set_xlabel("Time")
        g.set_ylabel("Value")
        g.legend()
        plt.show()

    def coverage(self) -> float:
        """
        Calculate the coverage of the confidence interval.
        """
        return np.mean((self.lower <= self._risk_seq) & (self._risk_seq <= self.upper))
    


class Hypothesis:
    def __init__(self, tolerance:float, lower_bound:ConfSeq, upper_bound:ConfSeq):
        self.tolerance = tolerance
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        pass

    def __call__(self, x:np.ndarray) -> bool:
        """
        Given a new input x, update the bounds and return weather the hypothesis is satisfied (within the tolerance).
        Parameters:
            x (np.ndarray): The input array.
        Returns:
            bool: The result of the function call.
        """
        raise NotImplementedError("Hypothesis class must implement __call__ method")


class Algorithm:
    """Algorithm
    This is a base class that represents an algorithm.
    It may be usefull to test multiple Hypothesis simultaneously.
    It may not be needed...

    parameters:
        hypothesis (dict[str, Hypothesis]): A dictionary of hypothesis to test.
    """
    def __init__(self, hypothesis:dict[str, Hypothesis]):
        self.hypothesis:dict[str, Hypothesis] = hypothesis

    def update(self, x:np.ndarray) -> bool:
        pass

    def __str__(self):
        return f"Algorithm: risk - {self.risk} and hypothesis - {self.hypothesis}"







