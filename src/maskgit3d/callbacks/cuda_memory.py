"""CUDA memory cleanup callback."""

import gc

import torch
from lightning.pytorch import Callback, LightningModule, Trainer


class CUDAMemoryCleanupCallback(Callback):
    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()
