"""Base class for LightningModule tasks."""

from typing import Any

from lightning import LightningModule


class BaseTask(LightningModule):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
