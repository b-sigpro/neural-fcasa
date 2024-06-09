from typing import Any

import torch

import lightning as lt


class CyclicAnnealerCallback(lt.Callback):
    def __init__(self, name: str, cycle: int, max_value: float, ini_period: int = 0, ini_max_value: float = 1.0):
        self.name = name
        self.cycle = cycle
        self.max_value = max_value

        self.ini_period = ini_period
        self.ini_max_value = ini_max_value

    def on_train_batch_start(self, trainer: lt.Trainer, pl_module: Any, batch: torch.Tensor, batch_idx: torch.Tensor):
        step = trainer.global_step / trainer.num_training_batches

        max_value = self.ini_max_value if step < self.ini_period else self.max_value
        setattr(pl_module, self.name, max_value * min(2 * (step % self.cycle) / self.cycle, 1))
