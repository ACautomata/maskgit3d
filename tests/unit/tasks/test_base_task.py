"""Tests for BaseTask."""

import pytest
import torch
import torch.nn as nn
from lightning import LightningModule

from src.maskgit3d.tasks.base_task import BaseTask


class SimpleModel(nn.Module):
    def __init__(self, in_features: int = 10, out_features: int = 2):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class SimpleTask(BaseTask):
    def __init__(self, model: nn.Module, lr: float = 1e-3):
        super().__init__()
        self.model = model
        self.lr = lr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        return nn.functional.mse_loss(y_hat, y)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


def test_base_task_inherits_lightning_module():
    model = SimpleModel()
    task = SimpleTask(model)

    assert isinstance(task, LightningModule)
    assert task.model is model


def test_base_task_save_hyperparameters():
    model = SimpleModel()
    task = SimpleTask(model, lr=1e-4)

    assert task.hparams.lr == 1e-4


def test_base_task_configure_optimizers():
    model = SimpleModel()
    task = SimpleTask(model, lr=1e-4)

    optimizers = task.configure_optimizers()
    assert isinstance(optimizers, torch.optim.Adam)
    assert optimizers.defaults["lr"] == 1e-4
