from dataclasses import dataclass, field
import numpy as np
import torch
import torch.nn as nn
from typing import Callable, NewType

from neural.module import LightningModel


class Risk:
    """Risk
    This is a base class that represents a risk function.
    """
    def __init__(self):
        pass

    @property
    def value(self):
        pass

class Bound:
    """Bound
    This is a base class that represents a bound function.
    """
    def __init__(self, bound_name:str, confidence:float, risk:Risk):
        self.bound_name = bound_name
        self.confidence = confidence
        self.risk = risk

        self._lower_bound:float = None
        self._upper_bound:float = None

    def _update_bounds(self, x:np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError("Bound class must implement _update_bounds method")

    def update(self, x:np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self._lower_bound, self._upper_bound = self._update_bounds(x)
        return self.lower, self.upper

    @property
    def lower(self):
        return self._lower_bound

    @property
    def upper(self):
        return self._upper_bound


class Hypothesis:
    def __init__(self, tolerance:float, lower_bound:Bound, upper_bound:Bound):
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







