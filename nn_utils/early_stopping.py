import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import wandb

from .metrics import SSIMLoss, mae, psnr, rmse

log = logging.getLogger(__name__)


class EarlyStopping(object):
    def __init__(self, step_interval: int, patience: int = 3, delta: float = 0.1):
        self.patience = patience
        self.delta = delta
        self.step_interval = step_interval
        self.current_step = 0
        self.counter = 0
        self.best_metric = None
        self.early_stop = False

    def step(self, metric: float):
        if self.current_step % self.step_interval == 0:
            if self.best_metric is None:
                self.best_metric = metric
            elif metric < self.best_metric + self.delta:
                self.counter += 1
                log.info(
                    f"EarlyStopping counter: {self.counter} out of {self.patience}"
                )
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_metric = metric
                self.counter = 0

        self.current_step += 1
