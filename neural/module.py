import os
import torch
from torch import Tensor, nn
from typing import Union, Sequence
from collections import defaultdict
from lightning import LightningModule
from torch import optim
from dataclasses import dataclass, field
from torch.utils.tensorboard import SummaryWriter


CLEAR_ML = False
try:
    from clearml import Task
    CLEAR_ML = True
except ImportError:
    pass

from utils.path_utils import join_paths
from utils.timing import Timer


ACTIVATIONS = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "softmax": nn.Softmax,
    "logsoftmax": nn.LogSoftmax,
    "lrelu": nn.LeakyReLU,
    "none": nn.Identity,
    None: nn.Identity,
}


# Default keyword arguments to pass to activation class constructors, e.g.
# activation_cls(**ACTIVATION_DEFAULT_KWARGS[name])
ACTIVATION_DEFAULT_KWARGS = defaultdict(
    dict,
    {
        ###
        "softmax": dict(dim=1),
        "logsoftmax": dict(dim=1),
    },
)



@dataclass
class EpochResult:
    loss: list[float] = field(default_factory=list)
    accuracy: list[float] = field(default_factory=list)
    forward_time: list[float] = field(default_factory=list)
    batch_sizes: list[float] = field(default_factory=list)
    key_suffix: str | None = field(default=None)

    def __post_init__(self):
        if self.key_suffix is not None:
            self.key_suffix = "/" + self.key_suffix
        else:
            self.key_suffix = ""

    def append(self, loss: float, accuracy: float, forward_time:float, batch_size: int) -> None:
        self.loss.append(loss)
        self.accuracy.append(accuracy)
        self.forward_time.append(forward_time)
        self.batch_sizes.append(batch_size)

    def clear(self) -> None:
        self.loss.clear()
        self.accuracy.clear()
        self.forward_time.clear()
        self.batch_sizes.clear()

    def make_result(self) -> dict:
        total = sum(self.batch_sizes)
        loss = torch.tensor(self.loss)
        accuracy = torch.tensor(self.accuracy)
        batch_sizes = torch.tensor(self.batch_sizes)

        results = {
            "loss" + self.key_suffix: (loss * batch_sizes).sum() / total,
            "accuracy" + self.key_suffix: (accuracy * batch_sizes).sum() / total,
            "forward_time" + self.key_suffix: sum(self.forward_time) / total,
        }

        return results


class LightningModel(nn.Module):
    def __init__(self, loss:nn.Module, name:str = None):
        super().__init__()
        self.name = name if name is not None else self.__class__.__name__
        self.loss = loss
        

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError
    
    def accuracy(self, preds: Tensor, labels: Tensor) -> Tensor:
        raise NotImplementedError
    

class LightningWrapper(LightningModule):

    def __init__(self, model:LightningModel, lr:float = 1e-3):
        super().__init__()
        self.name:str = model.name
        self.model:LightningModel = model
        self.epoch_results = dict(
            train=EpochResult(key_suffix="train"), 
            val=EpochResult(key_suffix="val")
        )

        self.loss:nn.Module = model.loss
        self.timer:Timer = Timer()
        self.lr = lr

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)

        lr_scheduler_config = {
            # REQUIRED: The scheduler instance
            "scheduler": optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.8, patience=3, min_lr=5e-5
            ),
            # The unit of the scheduler's step size, could also be 'step'.
            # 'epoch' updates the scheduler on epoch end whereas 'step'
            # updates it after a optimizer update.
            "interval": "epoch",
            # How many epochs/steps should pass between calls to
            # `scheduler.step()`. 1 corresponds to updating the learning
            # rate after every epoch/step.
            "frequency": 1,
            # Metric to to monitor for schedulers like `ReduceLROnPlateau`
            "monitor": "loss/val",
            # If set to `True`, will enforce that the value specified 'monitor'
            # is available when the scheduler is updated, thus stopping
            # training if not found. If set to `False`, it will only produce a warning
            "strict": False,
            # If using the `LearningRateMonitor` callback to monitor the
            # learning rate progress, this keyword can be used to specify
            # a custom logged name
            "name": 'lr',
        }
        return ([optimizer], [lr_scheduler_config])

    def on_train_epoch_start(self):  # TODO: batch size
        self.epoch_results["train"].clear()
        self.epoch_results["val"].clear()

        self.timer.start("train_epoch")
        if self.lr_schedulers() is not None:
            lr = self.optimizers().param_groups[0]["lr"]
            self.logger.log_metrics({"lr": lr}, step=self.current_epoch)
            # self.logger.log_metrics({"lr2": self.logger['lr']}, step=self.current_epoch)

    def on_train_epoch_end(self):
        epoch = self.current_epoch
        res = self.timer.stop("train_epoch")
        train_results = self.epoch_results["train"].make_result()
        val_results = self.epoch_results["val"].make_result()

        self.log_dict(train_results, logger=False)
        self.log_dict(val_results, logger=False)

        self.logger.log_metrics(train_results, step=epoch)
        self.logger.log_metrics(val_results, step=epoch)
        self.logger.log_metrics({"time/train_epoch": res}, step=epoch)
        # show results in progress bar
        self.log('train_loss', train_results["loss/train"], prog_bar=True, logger=False)
        self.log('val_loss', val_results["loss/val"], prog_bar=True, logger=False)
        self.log('train_acc', train_results["accuracy/train"], prog_bar=True, logger=False)
        self.log('val_acc', val_results["accuracy/val"], prog_bar=True, logger=False)

    def accuracy(self, preds: Tensor, labels: Tensor) -> Tensor:
        return self.model.accuracy(preds, labels)
        # class_preds = (preds > 0.5).to(torch.int32)
        # return (class_preds == labels).float().mean()

    def training_step(self, batch, batch_idx):
        self.timer.start("train_batch")

        data, labels = batch
        preds = self(data)
        loss = self.loss(preds, labels.to(dtype=torch.float))
        accuracy = self.accuracy(preds, labels)

        time_res = self.timer.stop("train_batch")
        
        self.epoch_results["train"].append(loss, accuracy, time_res, len(data))

        results = {"loss": loss, "accuracy": accuracy}
        return results

    def validation_step(self, batch, batch_idx):
        self.timer.start("val_batch")

        data, labels = batch
        preds = self.forward(data)
        # print(preds)
        loss = self.loss(preds, labels.to(dtype=torch.float))
        accuracy = self.accuracy(preds, labels)

        time_res = self.timer.stop("val_batch")

        self.epoch_results["val"].append(loss, accuracy, time_res, len(data))
        
        results = {"loss": loss, "accuracy": accuracy}
        return results



