from __future__ import annotations
from dataclasses import dataclass, field
from math import factorial
from torch.utils.data import DataLoader, Dataset
from lightning import Trainer, LightningModule
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.loggers.logger import DummyLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
CLEAR_ML = False
try:
    from clearml import Task
    CLEAR_ML = True
except ImportError:
    pass
# import wandb
from typing import Callable

from neural.config import DatasetConfig, TrainConfig, ExperimentConfig
import neural.config as config
from utils.path_utils import join_paths
from neural.module import LightningWrapper


class LightningTrainer:
    def __init__(
        self, experiment_config: ExperimentConfig, settings: config.TrainConfig
    ):
        self.exp_cfg = experiment_config
        self.settings = settings
        self.logger: TensorBoardLogger = DummyLogger()
        if CLEAR_ML:
            self.clearml_task: Task = None
        if self.settings.log:
            self.log_dir = join_paths(self.settings.logs_dir, self.exp_cfg.exp_name)
            self.logger = TensorBoardLogger(save_dir=self.log_dir)
            if CLEAR_ML:
                self.clearml_task = Task.init(project_name="MLnG - HW4", task_name=self.exp_cfg.exp_name)

    def fit(
        self,
        model: LightningWrapper,
        dl_train: DataLoader,
        dl_test: DataLoader,
    ) -> None:

        callbacks = []
        if self.settings.checkpoints:
            checkpoint_callback = ModelCheckpoint(
                dirpath=self.log_dir,
                filename=self.exp_cfg.exp_name,
                monitor="val_loss"
            )
            callbacks.append(checkpoint_callback)
        # callbacks.append(LearningRateMonitor('epoch'))

        trainer = Trainer(
            accelerator="auto",
            max_epochs=self.settings.num_epochs,
            logger=self.logger,
            log_every_n_steps=self.settings.log_every,
            max_time=self.settings.timeout,
            enable_checkpointing=self.settings.checkpoints,
            enable_progress_bar=True,
            enable_model_summary=True,
            callbacks=callbacks,
            # profiler='simple',
        )

        trainer.fit(model, dl_train, dl_test)

        if CLEAR_ML and self.settings.log:
            self.clearml_task.close()
        # wnb.finish()

    def clearml_add_tags(self, tags: list[str]) -> None:
        if CLEAR_ML and self.clearml_task is not None:
            self.clearml_task.add_tags(tags)

    def log_hyperparams(self, hparams: dict) -> None:
        if self.logger is not None:
            self.logger.log_hyperparams(hparams)
