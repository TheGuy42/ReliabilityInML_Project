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


# class Risk:
#     """Risk
#     This is a base class that represents a risk function.
#     """
#     def __init__(self):
#         pass

#     @property
#     def value(self):
#         pass

class ConfSeq:
    """
    This is a base class that represents a bound function.
    """
    def __init__(self, confidence:float, min_val:float=None, max_val:float=None, bound_name:str=None):
        self.name = bound_name if bound_name is not None else self.__class__.__name__
        self.conf_lvl = confidence
        self.min_val = min_val if min_val is not None else -np.inf
        self.max_val = max_val if max_val is not None else np.inf

        self._risk_seq:np.ndarray = None# np.ndarray([]).reshape(1)
        self._lower_cs:np.ndarray = None# np.ndarray([]).reshape(1)
        self._upper_cs:np.ndarray = None# np.ndarray([]).reshape(1)

    def calculate_cs(self, x:np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate the confidence interval bounds given a risk sequence x.
        """
        raise NotImplementedError("Bound class must implement _update_bounds method")

    def update(self, x:np.ndarray|float|None=None) -> tuple[np.ndarray, np.ndarray]:
        """
        update the confidence interval bounds given a new input x.
        """
        # if not isinstance(x, np.ndarray):
        #     x = np.asanyarray([x])
        # in_bounds = all(x >= self.min_val) and all(x <= self.max_val)
        # if not in_bounds:
        #     print(f"WARNING::Input x out of bounds [{self.min_val}, {self.max_val}], given x={x}")
        #     print(f"Clipping x to [{self.min_val}, {self.max_val}]")
        #     x = x.clip(self.min_val, self.max_val)
        
        # self._risk_seq = self._append_seq(x, self._risk_seq)
        if x is not None:
            self.append(x)
        
        if self._risk_seq is None:
            return None, None

        max_val = self.max_val if np.isfinite(self.max_val) else self._risk_seq.max()
        min_val = self.min_val if np.isfinite(self.min_val) else self._risk_seq.min()
        if max_val == min_val:
            max_val += 1

        normalized = (self._risk_seq - min_val) / (max_val - min_val)
        lower_cs, higher_cs = self.calculate_cs(normalized)

        self._lower_cs = lower_cs * (max_val - min_val) + min_val
        self._upper_cs = higher_cs * (max_val - min_val) + min_val

        return self.lower, self.upper
    
    def append(self, x:np.ndarray|float):
        """
        Append the input x to the risk sequence.
        """
        if not isinstance(x, np.ndarray):
            x = np.asanyarray([x])
        in_bounds = all(x >= self.min_val) and all(x <= self.max_val)
        if not in_bounds:
            print(f"WARNING::Input x out of bounds [{self.min_val}, {self.max_val}], given x={x}")
            print(f"Clipping x to [{self.min_val}, {self.max_val}]")
            x = x.clip(self.min_val, self.max_val)
        
        self._risk_seq = self._append_seq(x, self._risk_seq)

    def _append_seq(self, x:np.ndarray|float, seq:np.ndarray) -> np.ndarray:
        """
        append the input x to the sequence.
        parameters:
            x (np.ndarray): The input array of shape (n_samples) or a float if x is a single sample.
            seq (np.ndarray): The sequence to append to.
        """
        if isinstance(x, float):
            x = np.asanyarray([x])
        
        if seq is None:
            return x.reshape(-1)

        new_seq = np.concatenate((seq, x), axis=0)
        return new_seq

    @property
    def lower(self):
        return self._lower_cs

    @property
    def upper(self):
        return self._upper_cs
    
    def to_dataframe(self):
        df = pd.DataFrame(columns=["risk_seq", "lower_ci", "upper_ci"])
        df['time'] = np.arange(self._risk_seq.shape[0])
        df["risk_seq"] = self._risk_seq
        df["lower_ci"] = self._lower_cs
        df["upper_ci"] = self._upper_cs
        return df
    
    def plot(self):
        fig, axes = plt.subplots(1, 1, figsize=(12, 6))
        fig.set_dpi(100)

        df = self.to_dataframe()

        # g = sns.relplot(data=df, x='time', y=df.columns, hue=df.columns, ax=axes)
        g = sns.lineplot(data=df, x='time', y='risk_seq', ax=axes, label='Risk Sequence')
        g.fill_between(df['time'], df['lower_ci'], df['upper_ci'], alpha=0.35, color='red', label='Confidence Interval', zorder=10)
        g.set_title(f"{self.name} Confidence Interval; Confidence Level: {self.conf_lvl}; Coverage: {self.coverage()}")
        g.set_xlabel("Time")
        g.set_ylabel("Value")
        g.legend()
        plt.show()

    def coverage(self) -> float:
        """
        Calculate the coverage of the confidence interval over empirical mean of the risk.
        """
        emp_mean = self._risk_seq.cumsum() / np.arange(1, self._risk_seq.shape[0]+1)
        return np.mean((self.lower <= emp_mean) & (emp_mean <= self.upper))
    
    def reset(self):
        """
        Reset the confidence sequence and intervals.
        """
        self._risk_seq = None
        self._lower_cs = None
        self._upper_cs = None
    


class Hypothesis:
    def __init__(self, tolerance:float, lower_bound:ConfSeq, upper_bound:ConfSeq):
        self.tolerance = tolerance
        self.target_bound:ConfSeq = lower_bound
        self.source_bound:ConfSeq = upper_bound

        self.source_upper_cs = None
        self.target_lower_cs = None
        pass

    def calc_source_upper_cs(self, x:np.ndarray) -> np.ndarray:
        assert self.source_upper_cs is None, "Source upper bound was already calculated"
        self.source_upper_cs = self.source_bound.update(x)[1]
        return self.source_upper_cs

    def calc_target_lower_cs(self, x:np.ndarray) -> np.ndarray:
        lower_bound = self.target_bound.update(x)[0]
        self.target_lower_cs = lower_bound
        return self.target_lower_cs
        # if lower_bound is None:
        #     return self.target_lower_cs
    
        # if self.target_lower_cs is None:
        #     self.target_lower_cs = lower_bound[-1:]
        # else:
        #     self.target_lower_cs = np.concatenate((self.target_lower_cs, lower_bound[-1:]), axis=0)
        # return self.target_lower_cs

    @property
    def source_upper(self) -> np.ndarray:
        raise NotImplementedError("Hypothesis class must implement source_upper property")
    
    @property
    def target_lower(self) -> np.ndarray:
        raise NotImplementedError("Hypothesis class must implement target_lower property")
    
    def test(self, x:np.ndarray|None=None) -> bool:
        """
        Given a new input x, update the bounds and return weather the hypothesis is satisfied (within the tolerance).
        Parameters:
            x (np.ndarray): The input array or None if the bounds should be updated without adding new observations.
        Returns:
            bool: The result of the function call.
        """
        
        self.calc_target_lower_cs(x)
        if self.source_upper_cs is None or self.target_lower_cs is None:
            return True
        
        if self.target_lower[-1] > self.source_upper:
            return False
        return True
    
    def add_observations(self, x:np.ndarray|float, to_target:bool):
        """
        Add observations to the source or target bound.
        Parameters:
            x (np.ndarray|float): The new observations.
            to_target (bool): Add the observations to the target bound if True, else add to the source bound.
        """
        if to_target:
            self.target_bound.append(x)
        else:
            self.source_bound.append(x)
        
    
    def to_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame(columns=["risk_seq", "source_upper_bound", "target_lower_cs"])
        length = self.target_bound._risk_seq.shape[0]
        df['time'] = np.arange(length)
        df["risk_seq"] = self.target_bound._risk_seq
        df['empirical_target_mean'] = self.target_bound._risk_seq.cumsum() / np.arange(1, length+1)
        df['emp_source_mean'] = self.source_bound._risk_seq.mean()
        df["source_upper_bound"] = self.source_upper
        df["target_lower_cs"] = self.target_lower
        df["tol"] = self.tolerance
        df["rejected"] = ((df['target_lower_cs'] - df['source_upper_bound']) > 0).astype(bool)
        # source bound
        df['source_bound'] = self.source_bound.name
        df['source_confidence'] = self.source_bound.conf_lvl
        # target bound
        df['target_bound'] = self.target_bound.name
        df['target_confidence'] = self.target_bound.conf_lvl

        return df
    
    def plot(self, ax:plt.Axes=None, **kwargs) -> plt.Axes:
        if ax is None:
            fig, ax = plt.subplots(1, 1, dpi=200, **kwargs)
            # fig.set_dpi(200)

        df = self.to_dataframe()

        # g = sns.relplot(data=df, x='time', y=df.columns, hue=df.columns, ax=axes)
        g = sns.lineplot(data=df, x='time', y='risk_seq', ax=ax, label='Risk Sequence')

        diff =  (df['target_lower_cs'] - df['source_upper_bound'])
        g.fill_between(df['time'], df['target_lower_cs'], df['source_upper_bound'], where=diff > 0, alpha=0.35, color='green', label='Confidence Interval - H rejected', zorder=10)
        g.fill_between(df['time'], df['target_lower_cs'], df['source_upper_bound'], where=diff < 0, alpha=0.35, color='red', label='Confidence Interval - H holds', zorder=10)
        
        emp_source_mean = self.source_bound._risk_seq.mean()
        emp_target_mean = self.target_bound._risk_seq.cumsum() / np.arange(1, self.target_bound._risk_seq.shape[0]+1)
        sns.lineplot(x=df['time'], y=emp_target_mean, color='black', label='Empirical Target Mean', zorder=10, linestyle='--', ax=ax)
        g.hlines(emp_source_mean, 0, df['time'].max(), color='black', label='Empirical Source Mean', zorder=10, linestyle='-.')

        # g.set_title(f"Confidence Interval; Tolerance Level: {self.tolerance}")
        g.set_xlabel("Step")
        g.set_ylabel("Risk")
        # g.legend()
        # plt.show()
        return g

    def coverage(self) -> float:
        """
        Calculate the coverage of the confidence interval.
        """
        risk_seq = self.target_bound._risk_seq
        return np.mean((self.target_lower <= risk_seq) & (risk_seq <= self.source_upper))

    def reset(self, source:bool=True, target:bool=True):
        """
        Reset the confidence sequences of the source or target.
        """
        if source:
            self.source_bound.reset()
            self.source_upper_cs = None
        if target:
            self.target_bound.reset()
            self.target_lower_cs = None
    
    def set_bounds(self, min_val:float, max_val:float, source:bool=True, target:bool=True):
        if source:
            self.source_bound.max_val = max_val
            self.source_bound.min_val = min_val
        if target:
            self.target_bound.max_val = max_val
            self.target_bound.min_val = min_val










