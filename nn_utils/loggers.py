import logging
import pdb
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import wandb

from .metrics import SSIMLoss, mae, psnr, rmse
import csv, time

log = logging.getLogger(__name__)


def slice_figure(
    computed_slice, vmin: Optional[float] = 0, vmax: Optional[float] = 0.03
):
    fig, ax = plt.subplots(1, 1)
    if vmin is None and vmax is None:
        im = ax.imshow(computed_slice.cpu().detach().numpy(), cmap="gray")
    else:
        im = ax.imshow(
            computed_slice.cpu().detach().numpy(), vmin=vmin, vmax=vmax, cmap="gray"
        )
    ax.set_axis_off()
    fig.colorbar(im, ax=ax)

    return fig


class Logger(object):
    def __init__(self, log_steps, starting_step=0):
        self.log_steps = log_steps
        self.train_step_number = starting_step

        self.callbacks = {
            "volumes": self.log_volume,
            "loss": self.log_loss,
            "metrics": self.log_metrics,
        }

        self.artifacts_folder = Path.cwd() / "artifacts"
        self.artifacts_folder.mkdir(exist_ok=True)

        for callback_name in self.callbacks.keys():
            if callback_name not in self.log_steps.keys():
                raise ValueError(f"Callback {callback_name} not in log_steps")

        self.best_psnr = 0
        self.best_ssim = 0
        self.best_mae = np.inf
        self.best_rmse = np.inf

    @torch.inference_mode()
    def train_step(self, model, loss):
        for callback_name, num_steps in self.log_steps.items():
            if self.train_step_number % num_steps == 0:
                self.callbacks[callback_name](model, loss)
        self.train_step_number += 1

    @torch.inference_mode()
    def log_loss(self, model, loss, stage="train"):
        log_step_number = (
            self.train_step_number if stage == "train" else self.val_step_number
        )
        wandb.log({f"{stage}-loss": loss}, step=log_step_number)


class SingleCBCTLogger(Logger):
    def __init__(
        self,
        dataset,
        log_steps,
        starting_step=0,
        volume_id=0,
        volume_folder=None,
        tracking_metric="psnr",
    ):
        super().__init__(log_steps, starting_step)
        self.last_psnr = 0
        self.dataset = dataset
        self.tracking_metric = tracking_metric
        self.per_patient_metrics = {}
        self.start_time = time.time()
        self.volume_id = volume_id
        self.volume_folder = Path(volume_folder)

    @torch.inference_mode()
    def save_metrics(self, path):
        # save in a csv file
        with open(path, "w") as f:
            writer = csv.writer(f)
            writer.writerow(self.per_patient_metrics.keys())
            writer.writerows(zip(*self.per_patient_metrics.values()))

    def update_best_metric(
        self, metric, recon_volume, best_metric, metric_name, tracking_metric
    ):
        if metric > best_metric:
            best_metric = metric
            wandb.run.summary[f"best_{metric_name}"] = best_metric
            if tracking_metric == metric_name:
                self.store_volume(recon_volume, f"best_{metric_name}")

        return best_metric

    @torch.inference_mode()
    def log_metrics(self, model, loss, stage="train", store_volume=False):
        device = next(model.parameters()).device

        grid = torch.Tensor(self.dataset.reference_coords()).to(device)
        recon_volume = model(grid).view(grid.shape[0], grid.shape[1], grid.shape[2])
        gt_volume = torch.tensor(self.dataset.get_volume(), device=device)

        log.info(f"Data range: {torch.min(gt_volume)} - {torch.max(gt_volume)}")
        log.info(f"Recon range: {torch.min(recon_volume)} - {torch.max(recon_volume)}")

        data_range = torch.amax(gt_volume, dim=(0, 1, 2), keepdim=True)
        psnr_value = psnr(recon_volume[None, ...], gt_volume[None, ...]).cpu()
        ssim_value = SSIMLoss()(
            recon_volume[None, None, ...],
            gt_volume[None, None, ...],
            data_range[None, None, ...],
        )
        mae_value = mae(recon_volume, gt_volume)
        rmse_value = rmse(recon_volume, gt_volume)
        mae_value = mae_value.cpu()
        rmse_value = rmse_value.cpu().numpy()
        ssim_value = ssim_value.cpu().numpy()
        logging_step = self.train_step_number

        if "psnr" not in self.per_patient_metrics:
            self.per_patient_metrics["psnr"] = []
            self.per_patient_metrics["ssim"] = []
            self.per_patient_metrics["mae"] = []
            self.per_patient_metrics["rmse"] = []
            self.per_patient_metrics["patient_idx"] = []
            self.per_patient_metrics["step"] = []
            self.per_patient_metrics["time"] = []

        current_time = time.time()
        self.per_patient_metrics["psnr"].append(psnr_value)
        self.per_patient_metrics["ssim"].append(ssim_value)
        self.per_patient_metrics["mae"].append(mae_value)
        self.per_patient_metrics["rmse"].append(rmse_value)
        self.per_patient_metrics["patient_idx"].append(self.volume_id)
        self.per_patient_metrics["step"].append(logging_step)
        self.per_patient_metrics["time"].append(current_time - self.start_time)

        self.last_psnr = psnr_value

        self.best_psnr = self.update_best_metric(
            psnr_value, recon_volume, self.best_psnr, "psnr", self.tracking_metric
        )
        self.best_ssim = self.update_best_metric(
            ssim_value, recon_volume, self.best_ssim, "ssim", self.tracking_metric
        )
        self.best_mae = self.update_best_metric(
            mae_value, recon_volume, self.best_mae, "mae", self.tracking_metric
        )
        self.best_rmse = self.update_best_metric(
            rmse_value, recon_volume, self.best_rmse, "rmse", self.tracking_metric
        )

        wandb.log({f"{stage}-metrics/PSNR": psnr_value}, step=self.train_step_number)
        wandb.log({f"{stage}-metrics/SSIM": ssim_value}, step=self.train_step_number)
        wandb.log({f"{stage}-metrics/MAE": mae_value}, step=self.train_step_number)
        wandb.log({f"{stage}-metrics/RMSE": rmse_value}, step=self.train_step_number)
        log.info(f"{stage}-PSNR: {psnr_value}, {stage}-SSIM: {ssim_value}")

        if store_volume:
            grid = torch.Tensor(self.dataset.reference_coords()).to(device)
            recon_volume = model(grid).view(grid.shape[0], grid.shape[1], grid.shape[2])

            self.store_volume(
                recon_volume.detach().cpu(),
                Path(self.volume_folder)
                / f"{stage}_volume_{self.dataset.volume_id}.nii.gz",
            )

    @torch.no_grad()
    def store_volume(self, volume, path):
        # use numpy to store the volume
        np_volume = volume.cpu().detach().numpy()
        np.save(str(path), np_volume)

    @torch.inference_mode()
    def log_volume(self, model, loss):
        if not hasattr(self, "gt_logged"):
            self.gt_logged = False

        device = next(model.parameters()).device
        grid = torch.Tensor(self.dataset.reference_coords()).to(device)
        gt_volume = torch.tensor(self.dataset.get_volume(), device=device)
        vmin, vmax = gt_volume.min(), gt_volume.max()
        mid_slice = grid.shape[0] // 2
        im = model(
            grid[mid_slice, :, :],
        ).view(grid.shape[1], grid.shape[2])
        fig_axial = slice_figure(im, vmin=vmin, vmax=vmax)
        if not self.gt_logged:
            self.store_volume(gt_volume, "gt_volume")
            gt_axial = slice_figure(gt_volume[mid_slice, :, :], vmin=vmin, vmax=vmax)
        mid_slice = grid.shape[1] // 2
        im = model(
            grid[:, mid_slice, :],
        ).view(grid.shape[0], grid.shape[2])
        fig_coronal = slice_figure(im, vmin=vmin, vmax=vmax)
        if not self.gt_logged:
            gt_coronal = slice_figure(gt_volume[:, mid_slice, :], vmin=vmin, vmax=vmax)

        mid_slice = grid.shape[2] // 2
        im = model(
            grid[:, :, mid_slice],
        ).view(grid.shape[0], grid.shape[1])
        fig_saggital = slice_figure(im, vmin=vmin, vmax=vmax)
        if not self.gt_logged:
            gt_saggital = slice_figure(gt_volume[:, :, mid_slice], vmin=vmin, vmax=vmax)

        wandb.log(
            {
                f"axial/recon": fig_axial,
                f"coronal/recon": fig_coronal,
                f"saggital/recon": fig_saggital,
            },
            step=self.train_step_number,
        )
        if not self.gt_logged:
            wandb.log(
                {
                    f"axial/gt": gt_axial,
                    f"coronal/gt": gt_coronal,
                    f"saggital/gt": gt_saggital,
                },
                step=self.train_step_number,
            )
            self.gt_logged = True

        # close all plots at once
        plt.close("all")


class MultiCBCTLogger(Logger):
    def __init__(
        self,
        dataset,
        log_steps,
        val_log_steps,
        starting_step=0,
        metrics_folder=None,
        stage="train",
    ):
        super().__init__(log_steps, starting_step)
        self.val_step_number = 0
        self.dataset = dataset
        self.val_log_steps = val_log_steps
        self.starting_step = starting_step
        self.gt_logged = False
        self.stage = stage
        self.per_patient_metrics = {}
        self.metrics_folder = metrics_folder
        self.start_time = None

    @torch.inference_mode()
    def val_step(self, model, loss):
        if self.val_step_number == 0:
            self.start_time = time.time()
        for callback_name, num_steps in self.val_log_steps.items():
            if self.val_step_number % num_steps == 0:
                self.callbacks[callback_name](model, loss, stage=self.stage)
        self.val_step_number += 1

    @torch.inference_mode()
    def save_metrics(self, path):
        # save in a csv file
        with open(path, "w") as f:
            writer = csv.writer(f)
            writer.writerow(self.per_patient_metrics.keys())
            writer.writerows(zip(*self.per_patient_metrics.values()))

    @torch.inference_mode()
    def log_metrics(self, model, loss, stage="train", store_volume=False):
        current_time = time.time()
        logging_step = (
            self.train_step_number if stage == "train" else self.val_step_number
        )
        device = next(model.parameters()).device
        num_samples = self.dataset.num_patients
        selected_samples = torch.arange(num_samples).to(device)
        if self.dataset.volume_id is not None:
            selected_samples = [torch.tensor(self.dataset.volume_id, dtype=torch.long)]

        acc_psnr = []
        acc_ssim = []
        acc_mae = []
        acc_rmse = []

        if stage == "train":
            offset = 0
        elif stage == "val":
            offset = 200
        elif stage == "test":
            offset = 225
        for p in selected_samples:
            if self.dataset.volume_id is None:
                grid = torch.Tensor(self.dataset.reference_coords(p.item())).to(device)
                gt_volume = torch.tensor(
                    self.dataset.get_volume(p.item()), device=device
                )
            else:
                grid = torch.Tensor(self.dataset.reference_coords(0)).to(device)
                gt_volume = torch.tensor(self.dataset.get_volume(0), device=device)

            recon_volume = torch.zeros(
                grid.shape[0], grid.shape[1], grid.shape[2], device=device
            )
            for i in range(grid.shape[0]):
                recon_volume[i, :, :] = model(
                    grid[i, :, :], torch.tensor([p + offset])
                ).view(grid.shape[1], grid.shape[2])

            log.info(f"Data range: {torch.min(gt_volume)} - {torch.max(gt_volume)}")
            log.info(
                f"Recon range: {torch.min(recon_volume)} - {torch.max(recon_volume)}"
            )

            data_range = torch.amax(gt_volume, dim=(0, 1, 2), keepdim=True)
            psnr_value = psnr(recon_volume[None, ...], gt_volume[None, ...])
            ssim_value = SSIMLoss()(
                recon_volume[None, None, ...],
                gt_volume[None, None, ...],
                data_range[None, None, ...],
            )
            mae_value = mae(recon_volume, gt_volume)
            rmse_value = rmse(recon_volume, gt_volume)

            if "psnr" not in self.per_patient_metrics:
                self.per_patient_metrics["psnr"] = []
                self.per_patient_metrics["ssim"] = []
                self.per_patient_metrics["mae"] = []
                self.per_patient_metrics["rmse"] = []
                self.per_patient_metrics["patient_idx"] = []
                self.per_patient_metrics["step"] = []
                self.per_patient_metrics["time"] = []

            self.per_patient_metrics["psnr"].append(psnr_value.cpu().numpy())
            self.per_patient_metrics["ssim"].append(ssim_value.cpu().numpy())
            self.per_patient_metrics["mae"].append(mae_value.cpu().numpy())
            self.per_patient_metrics["rmse"].append(rmse_value.cpu().numpy())
            self.per_patient_metrics["patient_idx"].append(p.item())
            self.per_patient_metrics["step"].append(logging_step)
            self.per_patient_metrics["time"].append(current_time - self.start_time)

            acc_mae.append(mae_value.cpu())
            acc_rmse.append(rmse_value.cpu().numpy())
            acc_psnr.append(psnr_value.cpu().numpy())
            acc_ssim.append(ssim_value.cpu().numpy())

        # dump the per-patient metrics to a csv file
        if self.metrics_folder is not None:
            path_to_csv = Path(self.metrics_folder) / f"{stage}_per_patient_metrics.csv"
            with open(path_to_csv, "w") as f:
                writer = csv.writer(f)
                writer.writerow(self.per_patient_metrics.keys())
                writer.writerows(zip(*self.per_patient_metrics.values()))
        if self.dataset.volume_id is not None:
            stage = f"{stage}_{self.dataset.volume_id}"
        if store_volume:
            for p in selected_samples:
                if self.dataset.volume_id is None:
                    grid = torch.Tensor(self.dataset.reference_coords(p.item())).to(
                        device
                    )
                else:
                    grid = torch.Tensor(self.dataset.reference_coords(0)).to(device)
                recon_volume = torch.zeros(
                    grid.shape[0], grid.shape[1], grid.shape[2], device=device
                )
                for i in range(grid.shape[0]):
                    recon_volume[i, :, :] = model(
                        grid[i, :, :], torch.tensor([p + offset])
                    ).view(grid.shape[1], grid.shape[2])
                self.store_volume(
                    recon_volume,
                    Path(self.metrics_folder) / f"{stage}_volume_{p}.nii.gz",
                )

        if np.mean(acc_psnr) > self.best_psnr:
            self.best_psnr = np.mean(acc_psnr)
            wandb.run.summary[f"{stage}-best_psnr"] = self.best_psnr
        if np.mean(acc_ssim) > self.best_ssim:
            self.best_ssim = np.mean(acc_ssim)
            wandb.run.summary[f"{stage}-best_ssim"] = self.best_ssim
        if np.mean(acc_mae) < self.best_mae:
            self.best_mae = np.mean(acc_mae)
            wandb.run.summary[f"{stage}-best_mae"] = self.best_mae
        if np.mean(acc_rmse) < self.best_rmse:
            self.best_rmse = np.mean(acc_rmse)
            wandb.run.summary[f"{stage}-best_rmse"] = self.best_rmse

        wandb.log(
            {f"{stage}-metrics/PSNR": np.mean(acc_psnr)},
            step=self.starting_step + logging_step,
        )
        wandb.log(
            {f"{stage}-metrics/SSIM": np.mean(acc_ssim)},
            step=self.starting_step + logging_step,
        )
        wandb.log(
            {f"{stage}-metrics/MAE": np.mean(acc_mae)},
            step=self.starting_step + logging_step,
        )
        wandb.log(
            {f"{stage}-metrics/RMSE": np.mean(acc_rmse)},
            step=self.starting_step + logging_step,
        )
        log.info(
            f"{stage}-PSNR: {np.mean(acc_psnr)}, {stage}-SSIM: {np.mean(acc_ssim)}"
        )

    @torch.inference_mode()
    def store_volume(self, volume: torch.Tensor, path: Path):
        np.save(str(path), volume.cpu().numpy())

    @torch.inference_mode()
    def log_volume(self, model, loss, stage="train"):
        if not hasattr(self, "gt_logged"):
            self.gt_logged = False
        logging_step = (
            self.train_step_number if stage == "train" else self.val_step_number
        )
        num_samples = min(self.dataset.num_patients, 5)
        device = next(model.parameters()).device
        selected_samples = torch.arange(num_samples).to(device)
        if self.dataset.volume_id is not None:
            selected_samples = [torch.tensor(self.dataset.volume_id, dtype=torch.long)]

        if stage == "train":
            offset = 0
        elif stage == "val":
            offset = 200
        elif stage == "test":
            offset = 225
        else:
            raise ValueError(f"Unknown stage {stage}")

        for p in selected_samples:
            log.info(f"Processing patient {p} with offset {offset}")
            if self.dataset.volume_id is None:
                grid = torch.Tensor(self.dataset.reference_coords(p.item())).to(device)
                gt_volume = torch.tensor(
                    self.dataset.get_volume(p.item()), device=device
                )
            else:
                grid = torch.Tensor(self.dataset.reference_coords(0)).to(device)
                gt_volume = torch.tensor(self.dataset.get_volume(0), device=device)
            # grid = torch.Tensor(self.dataset.reference_coords(p.item())).to(device)
            # # gt_volume = torch.tensor(self.dataset.get_volume(p.item())[1], device=device).view(self.dataset.volume_shapes[p])
            # gt_volume = torch.tensor(self.dataset.get_volume(p.item()), device=device)

            mid_slice = grid.shape[0] // 2
            im = model(grid[mid_slice, :, :], torch.tensor([p + offset])).view(
                grid.shape[1], grid.shape[2]
            )
            fig_axial = slice_figure(im, vmin=None, vmax=None)
            if not self.gt_logged:
                gt_axial = slice_figure(
                    gt_volume[mid_slice, :, :], vmin=None, vmax=None
                )
            mid_slice = grid.shape[1] // 2
            im = model(grid[:, mid_slice, :], torch.tensor([p + offset])).view(
                grid.shape[0], grid.shape[2]
            )
            fig_coronal = slice_figure(im, vmin=None, vmax=None)
            if not self.gt_logged:
                gt_coronal = slice_figure(
                    gt_volume[:, mid_slice, :], vmin=None, vmax=None
                )

            mid_slice = grid.shape[2] // 2
            im = model(grid[:, :, mid_slice], torch.tensor([p + offset])).view(
                grid.shape[0], grid.shape[1]
            )
            fig_saggital = slice_figure(im, vmin=None, vmax=None)
            if not self.gt_logged:
                gt_saggital = slice_figure(
                    gt_volume[:, :, mid_slice], vmin=None, vmax=None
                )

            wandb.log(
                {
                    f"{stage}-axial/recon_{p + offset}": fig_axial,
                    f"{stage}-coronal/recon_{p + offset}": fig_coronal,
                    f"{stage}-saggital/recon_{p + offset}": fig_saggital,
                },
                step=self.starting_step + logging_step,
            )
            plt.close(fig_axial)
            plt.close(fig_coronal)
            plt.close(fig_saggital)

            if not self.gt_logged:
                wandb.log(
                    {
                        f"{stage}-axial/gt": gt_axial,
                        f"{stage}-coronal/gt": gt_coronal,
                        f"{stage}-saggital/gt": gt_saggital,
                    },
                    step=self.starting_step + logging_step,
                )
                plt.close(gt_axial)
                plt.close(gt_coronal)
                plt.close(gt_saggital)
                self.gt_logged = True

        # close all plots at once
        plt.close("all")


class MultiCBCTReconLogger(Logger):
    def __init__(self, dataset, log_steps, starting_step=0, val_log_steps=None):
        super().__init__(log_steps, starting_step)
        self.val_step_number = 0
        self.starting_step = starting_step
        self.dataset = dataset
        self.val_log_steps = val_log_steps
        self.gt_logged = False
        self.callbacks["gradients"] = self.log_gradients

    def log_gradients(self, model, loss, stage="train"):
        # Compute the average gradient norm for the whole codebook
        norm_grads = []
        for params in model.conditioning.parameters():
            if params.grad is not None:
                norm_grads.append(params.grad.norm())
        norm_grads = torch.stack(norm_grads).mean()

        wandb.log({"norm_grads_conditioning": norm_grads}, step=self.train_step_number)

        # Compute gradients from the rest of the network
        norm_grads = []
        for params in model.parameters():
            if params.grad is not None:
                norm_grads.append(params.grad.norm())
        norm_grads = torch.stack(norm_grads).mean()

        wandb.log({"norm_grads": norm_grads}, step=self.train_step_number)

    @torch.inference_mode()
    def val_step(self, model, loss):
        for callback_name, num_steps in self.val_log_steps.items():
            if self.val_step_number % num_steps == 0:
                self.callbacks[callback_name](model, loss, stage="val")
        self.val_step_number += 1

    @torch.inference_mode()
    def log_metrics(self, model, loss, stage="train"):
        logging_step = (
            self.train_step_number if stage == "train" else self.val_step_number
        )
        device = next(model.parameters()).device
        num_samples = min(self.dataset.num_patients, 5)
        selected_samples = torch.arange(num_samples).to(device)

        acc_psnr = []
        acc_ssim = []
        acc_mae = []
        acc_rmse = []

        for p in selected_samples:
            grid, gt_volume, global_patient_idx = self.dataset.get_volume(p.item())
            grid = torch.Tensor(
                np.reshape(grid, (*self.dataset.volume_shapes[p], 3))
            ).to(device)
            gt_volume = torch.tensor(gt_volume, device=device).view(
                self.dataset.volume_shapes[p]
            )

            recon_volume = torch.zeros(
                grid.shape[0], grid.shape[1], grid.shape[2], device=device
            )
            for i in range(grid.shape[0]):
                recon_volume[i, :, :] = model(
                    grid[i, :, :], torch.tensor([global_patient_idx])
                ).view(grid.shape[1], grid.shape[2])

            log.info(f"Data range: {torch.min(gt_volume)} - {torch.max(gt_volume)}")
            log.info(
                f"Recon range: {torch.min(recon_volume)} - {torch.max(recon_volume)}"
            )

            data_range = torch.amax(gt_volume, dim=(0, 1, 2), keepdim=True)
            psnr_value = psnr(recon_volume[None, ...], gt_volume[None, ...]).cpu()
            ssim_value = SSIMLoss()(
                recon_volume[None, None, ...],
                gt_volume[None, None, ...],
                data_range[None, None, ...],
            )
            mae_value = mae(recon_volume, gt_volume)
            rmse_value = rmse(recon_volume, gt_volume)
            acc_mae.append(mae_value.cpu())
            acc_rmse.append(rmse_value.cpu().numpy())
            acc_psnr.append(psnr_value)
            acc_ssim.append(ssim_value.cpu().numpy())

        if np.mean(acc_psnr) > self.best_psnr:
            self.best_psnr = np.mean(acc_psnr)
            wandb.run.summary[f"{stage}-best_psnr"] = self.best_psnr
        if np.mean(acc_ssim) > self.best_ssim:
            self.best_ssim = np.mean(acc_ssim)
            wandb.run.summary[f"{stage}-best_ssim"] = self.best_ssim
        if np.mean(acc_mae) < self.best_mae:
            self.best_mae = np.mean(acc_mae)
            wandb.run.summary[f"{stage}-best_mae"] = self.best_mae
        if np.mean(acc_rmse) < self.best_rmse:
            self.best_rmse = np.mean(acc_rmse)
            wandb.run.summary[f"{stage}-best_rmse"] = self.best_rmse

        wandb.log({f"{stage}-metrics/PSNR": np.mean(acc_psnr)}, step=logging_step)
        wandb.log({f"{stage}-metrics/SSIM": np.mean(acc_ssim)}, step=logging_step)
        wandb.log({f"{stage}-metrics/MAE": np.mean(acc_mae)}, step=logging_step)
        wandb.log({f"{stage}-metrics/RMSE": np.mean(acc_rmse)}, step=logging_step)
        log.info(
            f"{stage}-PSNR: {np.mean(acc_psnr)}, {stage}-SSIM: {np.mean(acc_ssim)}"
        )

    @torch.inference_mode()
    def log_volume(self, model, loss, stage="train"):
        if not hasattr(self, "gt_logged"):
            self.gt_logged = False

        logging_step = (
            self.train_step_number if stage == "train" else self.val_step_number
        )
        num_samples = min(self.dataset.num_patients, 5)
        selected_samples = torch.arange(num_samples)
        device = next(model.parameters()).device

        for p in selected_samples:
            grid, gt_volume, global_patient_idx = self.dataset.get_volume(p.item())
            grid = torch.Tensor(
                np.reshape(grid, (*self.dataset.volume_shapes[p], 3))
            ).to(device)
            gt_volume = torch.tensor(gt_volume, device=device).view(
                self.dataset.volume_shapes[p]
            )

            mid_slice = grid.shape[0] // 2
            im = model(grid[mid_slice, :, :], torch.tensor([global_patient_idx])).view(
                grid.shape[1], grid.shape[2]
            )
            fig_axial = slice_figure(im, vmin=None, vmax=None)
            if not self.gt_logged:
                gt_axial = slice_figure(
                    gt_volume[mid_slice, :, :], vmin=None, vmax=None
                )
            mid_slice = grid.shape[1] // 2
            im = model(grid[:, mid_slice, :], torch.tensor([global_patient_idx])).view(
                grid.shape[0], grid.shape[2]
            )
            fig_coronal = slice_figure(im, vmin=None, vmax=None)
            if not self.gt_logged:
                gt_coronal = slice_figure(
                    gt_volume[:, mid_slice, :], vmin=None, vmax=None
                )

            mid_slice = grid.shape[2] // 2
            im = model(grid[:, :, mid_slice], torch.tensor([global_patient_idx])).view(
                grid.shape[0], grid.shape[1]
            )
            fig_saggital = slice_figure(im, vmin=None, vmax=None)
            if not self.gt_logged:
                gt_saggital = slice_figure(
                    gt_volume[:, :, mid_slice], vmin=None, vmax=None
                )

            wandb.log(
                {
                    f"{stage}-axial/recon_{global_patient_idx}": fig_axial,
                    f"{stage}-coronal/recon_{global_patient_idx}": fig_coronal,
                    f"{stage}-saggital/recon_{global_patient_idx}": fig_saggital,
                },
                step=logging_step,
            )
            if not self.gt_logged:
                log.info("Logging ground truth")
                wandb.log(
                    {
                        f"{stage}-axial/gt_{global_patient_idx}": gt_axial,
                        f"{stage}-coronal/gt_{global_patient_idx}": gt_coronal,
                        f"{stage}-saggital/gt_{global_patient_idx}": gt_saggital,
                    },
                    step=logging_step,
                )
            # close all plots at once
            plt.close("all")

        if not self.gt_logged:
            self.gt_logged = True
