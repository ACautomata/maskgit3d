"""Base class for LightningModule tasks."""

from typing import Any

from lightning import LightningModule


class BaseTask(LightningModule):
    """Base class for LightningModule tasks.

    Provides a common base for all training tasks in the project with
    standard configuration for hyperparameter saving.

    Args:
        *args: Positional arguments passed to LightningModule.
        **kwargs: Keyword arguments passed to LightningModule.

    Attributes:
        Automatically saves hyperparameters via save_hyperparameters().
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
