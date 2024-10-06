import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from algorithm import ConfSeq
from risk import Risk


class Hypothesis:
    def __init__(
        self,
        risk: Risk,
        tolerance: float,
        lower_bound: ConfSeq,
        upper_bound: ConfSeq,
    ):
        self.risk: Risk = risk
        self.tolerance = tolerance
        self.target_bound: ConfSeq = lower_bound
        self.source_bound: ConfSeq = upper_bound

        self.source_upper_cs = None
        self.target_lower_cs = None

        self.input_seq = None
        self.label_seq = None
        self.pred_seq = None

        self.risk_seq = None
        pass

    def calc_source_upper_cs(self, x: np.ndarray) -> np.ndarray:
        assert self.source_upper_cs is None, "Source upper bound was already calculated"
        self.source_upper_cs = self.source_bound.update(x)[1]
        return self.source_upper_cs

    def calc_target_lower_cs(self, x: np.ndarray) -> np.ndarray:
        self.target_lower_cs = self.target_bound.update(x)[0]
        return self.target_lower_cs

    @property
    def source_upper(self) -> np.ndarray:
        raise NotImplementedError(
            "Hypothesis class must implement source_upper property"
        )

    @property
    def target_lower(self) -> np.ndarray:
        raise NotImplementedError(
            "Hypothesis class must implement target_lower property"
        )

    def test(self, x: np.ndarray) -> bool:
        """
        Given a new input x, update the bounds and return weather the hypothesis is satisfied (within the tolerance).
        Parameters:
            x (np.ndarray): The input array.
        Returns:
            bool: The result of the function call.
        """
        self.calc_target_lower_cs(x)
        if self.target_lower[-1] > self.source_upper:
            return False
        return True

    def to_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame(columns=["data", "source_upper_bound", "target_lower_cs"])
        length = self.target_bound._risk_seq.shape[0]
        df["time"] = np.arange(length)
        df["risk_seq"] = self.target_bound._risk_seq
        df["source_upper_bound"] = self.source_upper
        df["target_lower_cs"] = self.target_lower
        df["tol"] = self.tolerance
        df["rejected"] = (
            (df["target_lower_cs"] - df["source_upper_bound"]) > 0
        ).astype(bool)
        # source bound
        df["source_bound"] = self.source_bound.name
        df["source_confidence"] = self.source_bound.conf_lvl
        # target bound
        df["target_bound"] = self.target_bound.name
        df["target_confidence"] = self.target_bound.conf_lvl

        return df

    def plot(self):
        fig, axes = plt.subplots(1, 1, figsize=(12, 6))
        fig.set_dpi(200)

        df = self.to_dataframe()

        # g = sns.relplot(data=df, x='time', y=df.columns, hue=df.columns, ax=axes)
        g = sns.lineplot(data=df, x="time", y="risk_seq", ax=axes, label="Risk Sequence")

        diff = df["target_lower_cs"] - df["source_upper_bound"]
        g.fill_between(
            df["time"],
            df["target_lower_cs"],
            df["source_upper_bound"],
            where=diff > 0,
            alpha=0.35,
            color="green",
            label="Confidence Interval - H rejected",
            zorder=10,
        )
        g.fill_between(
            df["time"],
            df["target_lower_cs"],
            df["source_upper_bound"],
            where=diff < 0,
            alpha=0.35,
            color="red",
            label="Confidence Interval - H holds",
            zorder=10,
        )

        emp_source_mean = self.source_bound._risk_seq.mean()
        emp_target_mean = self.target_bound._risk_seq.cumsum() / np.arange(
            1, self.target_bound._risk_seq.shape[0] + 1
        )
        sns.lineplot(
            x=df["time"],
            y=emp_target_mean,
            color="black",
            label="Empirical Target Mean",
            zorder=10,
            linestyle="--",
            ax=axes,
        )
        g.hlines(
            emp_source_mean,
            0,
            df["time"].max(),
            color="black",
            label="Empirical Source Mean",
            zorder=10,
            linestyle="-.",
        )

        g.set_title(f"Confidence Interval; Tolerance Level: {self.tolerance}")
        g.set_xlabel("Time")
        g.set_ylabel("Risk")
        g.legend()
        plt.show()

    def coverage(self) -> float:
        """
        Calculate the coverage of the confidence interval.
        """
        risk_seq = self.target_bound._risk_seq
        return np.mean(
            (self.target_lower <= risk_seq) & (risk_seq <= self.source_upper)
        )


class Algorithm:
    """Algorithm
    This is a base class that represents an algorithm.
    It may be usefull to test multiple Hypothesis simultaneously.
    It may not be needed...

    parameters:
        hypothesis (dict[str, Hypothesis]): A dictionary of hypothesis to test.
    """

    def __init__(self, hypothesis: dict[str, Hypothesis]):
        self.hypothesis: dict[str, Hypothesis] = hypothesis

    def update(self, x: np.ndarray) -> bool:
        pass
