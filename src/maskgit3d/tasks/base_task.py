"""Base class for LightningModule tasks."""

from collections.abc import Mapping
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
        self._callback_payloads: dict[str, dict[str, Any]] = {}
        self.save_hyperparameters()

    def save_callback_payload(self, stage: str, payload: Mapping[str, Any]) -> None:
        self._callback_payloads[stage] = dict(payload)

    def get_callback_payload(self, stage: str) -> dict[str, Any] | None:
        payload = self._callback_payloads.get(stage)
        if payload is None:
            return None
        return dict(payload)

    def pop_callback_payload(self, stage: str) -> dict[str, Any] | None:
        return self._callback_payloads.pop(stage, None)
