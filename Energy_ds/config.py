
import pandas as pd
import numpy as np
from dataclasses import dataclass, fields
from typing import Callable
from enum import Enum

class REGION(Enum):
    AEP = 1
    COMED = 2
    DAYTON = 3
    DEOK = 4
    DOM = 5
    DUQ = 6
    EKPC = 7
    FE = 8
    PJME = 9
    PJMW = 10
    NI = 11

    @staticmethod
    def from_str(region: str):
        return REGION.__dict__[region]

class SEASON(Enum):
    SPRING = 1
    SUMMER = 2
    FALL = 3
    WINTER = 4

    @staticmethod
    def from_str(season: str):
        return SEASON.__dict__[season.upper()]


@dataclass
class DatasetConfig:
    past_window: int = 0 # number of past hours included with each sample
    region: list[REGION] = None # region to include in the dataset
    years: tuple[int, int] = None # start and end years to include in the dataset [)
    season: list[SEASON]|SEASON = None # season to include in the dataset
    months: list[int] = None # months to include in the dataset
    days: list[int] = None # days of the week to include in the dataset

    def __post_init__(self):
        if isinstance(self.season, SEASON):
            self.season = [self.season]

    @property
    def conditions(self):
        conds = []
        if self.region is not None:
            conds.append(region_cond(self.region))
        if self.years is not None:
            conds.append(year_cond(*self.years))
        if self.season is not None:
            conds.append(season_cond(self.season))
        if self.months is not None:
            conds.append(month_cond(self.months))
        if self.days is not None:
            conds.append(day_cond(self.days))

        return cond_and(conds)

def year_cond(start:int, end:int):
    return lambda data: data['Year'].between(start, end, inclusive='left')

def season_cond(seasons:list[SEASON]):
    return lambda data: data['Season'].isin([s.value for s in seasons])

def month_cond(months:list[int]):
    return lambda data: data['Month'].isin(months)

def day_cond(days:list[int]):
    return lambda data: data['Day'].isin(days)

def cond_and(conds:list):
    def cond(data):
        mask = pd.Series(len(data)*[True])
        for c in conds:
            mask = mask & c(data)
        return mask
    # return lambda data: ([cond(data) for cond in conds])
    return cond

def region_cond(regions:list[REGION]):
    return lambda data: data['Region'].isin([r.value for r in regions])











