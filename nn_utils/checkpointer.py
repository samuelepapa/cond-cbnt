import logging
from pathlib import Path
from typing import Union

import torch
import torch.nn as nn

log = logging.getLogger(__name__)


class CheckpointMaker:
    def __init__(self, path: Union[str, Path], num_steps: int):
        self.num_steps = num_steps
        self.current_step = 0
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)

        self.best_metric = float("inf")

    def step(self, model, optimizer, loss, metric=None):
        if self.current_step % self.num_steps == 0:
            self.save_checkpoint(model, optimizer, loss, metric)
        self.current_step += 1

    def save_checkpoint(self, model, optimizer, loss, metric=None):
        """
        Save a checkpoint to the given path
        Args:
            model: the model to save
            optimizer: the optimizer to save
            epoch: the epoch to save
            loss:
            metric:

        Returns:

        """
        checkpoint = {
            "step": self.current_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        }
        # torch.save(checkpoint, self.path / f"checkpoint_{self.current_step}.pt")
        # # save the latest checkpoint
        # torch.save(checkpoint, self.path / "checkpoint.pt")
        #
        # log.info(
        #     f"Checkpoint saved at {self.path / f'checkpoint_{self.current_step}.pt'} and {self.path / 'checkpoint.pt'}"
        # )

        if metric is not None:
            if metric < self.best_metric:
                torch.save(checkpoint, self.path / "best_checkpoint.pt")
                self.best_metric = metric
                log.info(f"Best checkpoint saved at {self.path / 'best_checkpoint.pt'}")

    def load_best_checkpoint(self, model):
        checkpoint_path = self.path / "best_checkpoint.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint {checkpoint_path} does not exist")

        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        # optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        # epoch = checkpoint["epoch"]

        return model  # , epoch

    def load_checkpoint(self, model, optimizer, step=None):
        """
        Load a checkpoint from the given path

        Args:
            model: the model to load the checkpoint into
            optimizer: the optimizer to load the checkpoint into
            step: the epoch to load the checkpoint from. If None, load the latest checkpoint

        Returns:
            model, optimizer, epoch
        """

        if step is None:
            checkpoint_path = self.path / "checkpoint.pt"
        else:
            checkpoint_path = self.path / f"checkpoint_{step}.pt"

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint {checkpoint_path} does not exist")

        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint["epoch"]

        return model, optimizer, epoch
