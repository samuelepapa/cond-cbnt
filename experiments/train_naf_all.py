import logging
import os
import time
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.cuda
import torch.nn.functional as F
import wandb
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.naf_dataset import SingleVolume
from nef.instance.hashnet import HashNet
from nn_utils.checkpointer import CheckpointMaker
from nn_utils.early_stopping import EarlyStopping
from nn_utils.loggers import SingleCBCTLogger

log = logging.getLogger(__name__)
import pdb


@hydra.main(config_path="../config", config_name="multi_volume.yaml", version_base=None)
def main(cfg):
    experiment_path = Path(os.getcwd())
    log.info("Experiment path: {}".format(experiment_path))
    # We possibly want to add fields to the config file. Thus, we set struct to False.
    OmegaConf.set_struct(cfg, False)

    # set all the seeds
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)

    wandb.login(key="da05829c15c052ce21ea676a2050405df8abf981")
    if cfg.stage == "valid":
        volume_ids = range(200, 225)
    elif cfg.stage == "test":
        volume_ids = range(225, 250)
    else:
        raise ValueError("Invalid stage")

    for volume_id in volume_ids:
        perform_training(cfg, volume_id)


def perform_training(cfg, volume_id):
    output_dir = Path(os.getcwd()) / cfg.run_name
    output_dir.mkdir(exist_ok=True)
    # Initialize logging.
    wandb.init(
        project=cfg.project_name,
        group=cfg.stage + "_" + cfg.run_name,
        name=cfg.run_name + "_" + str(volume_id),
        mode="online" if cfg.wandb_log else "offline",
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    # Construct datamodule
    log.info("Loading dataset")

    # Construct neural field model
    model = HashNet(
        num_in=3,  # train_dataset.dim,
        num_hidden_in=cfg.nef.num_hidden,
        num_hidden_out=cfg.nef.num_hidden,
        num_layers=cfg.nef.num_layers,
        num_out=1,  # train_dataset.channels,
        num_levels=cfg.nef.hash.num_levels,
        level_dim=cfg.nef.hash.level_dim,
        base_resolution=cfg.nef.hash.base_resolution,
        log2_max_params_per_level=cfg.nef.hash.log2_max_params_per_level,
        final_act=cfg.nef.final_act,
        skip_conn=cfg.nef.skip_conn,
    )
    model.to("cuda")
    log.info(f"Memory allocated: {torch.cuda.memory_allocated()}")
    pdb.set_trace()
    train_dataset = SingleVolume(
        volumes_dir=cfg.dataset.path,
        num_steps=cfg.dataset.num_steps,
        projs_sample_step=cfg.dataset.projs_sample_step,
        num_projs=cfg.dataset.num_projs,
        norm_const=cfg.dataset.norm_const,
        name=cfg.dataset.name,
        num_rays=cfg.dataset.num_rays,
        noisy_projections=cfg.dataset.noisy_projections,
        volume_id=volume_id,
    )

    # Construct dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=0,  # has to be 0 because we load everything in GPU already
        pin_memory=False,
    )

    # Get device
    device = torch.device("cuda") if (cfg.cuda and torch.cuda.is_available()) else "cpu"
    if not cfg.cuda:
        raise ValueError(
            "Only cuda is supported at the moment for performance purposes."
        )

    log.info(f"Using device: {device}.")
    log.info(f"Number of parameters: {sum([p.numel() for p in model.parameters()])}. ")

    model = model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        weight_decay=cfg.optimizer.weight_decay,
        lr=cfg.training.lr,
        betas=cfg.optimizer.betas,
        eps=cfg.optimizer.eps,
    )
    # Scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=cfg.training.lr_step_size, gamma=cfg.training.lr_gamma
    )

    logger = SingleCBCTLogger(train_dataset, cfg.log_steps, volume_folder=output_dir)

    # Early stopping
    early_stopping = EarlyStopping(
        patience=cfg.training.patience,
        delta=cfg.training.delta,
        step_interval=cfg.training.early_stopping_step_interval,
    )

    start_time = time.time()

    pbar = tqdm(total=cfg.training.epochs * len(train_loader))
    logging_time = 0
    log.info(f"GPU memory: {torch.cuda.memory_allocated()}")
    memory_file = output_dir / f"memory_{volume_id}.txt"
    memory_file.write_text(f"GPU memory: {torch.cuda.memory_allocated()/1024**3} GB")
    for epoch in range(1, cfg.training.epochs):
        interrupted = train(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            device=device,
            logger=logger,
            scheduler=scheduler,
            start_time=start_time,
            max_time=cfg.training.max_time,
            pbar=pbar,
            early_stopping=early_stopping,
            logging_time=logging_time,
            max_iter=cfg.training.max_iter,
        )
        if interrupted:
            break

    logger.save_metrics(output_dir / f"metrics_{volume_id}.csv")
    wandb.finish()

    return logger.best_psnr, logger.best_ssim, logger.best_mae, logger.best_rmse


def process_batch(data, dset, device, extent):
    target_pixels, sources, rays, vol_bboxes, patient_idx = data
    # move all data to the device
    vol_bboxes = vol_bboxes.to(device)

    # find length of longest diagonal, this can be calculated once for the whole volume
    max_box_size = torch.sqrt(
        torch.sum(torch.pow(vol_bboxes[:, :, 1] - vol_bboxes[:, :, 0], 2))
    )
    tmin = dset.geometry.source_to_center_dst - max_box_size / 2
    tmax = dset.geometry.source_to_center_dst + max_box_size / 2

    # sample the points along the rays
    samples = (
        torch.linspace(0, 1, dset.num_steps, device=device)[None, None, :]
        * (tmax - tmin)[..., None]
        + tmin[..., None]
    )

    samples = samples.expand([rays.shape[0], rays.shape[-2], dset.num_steps])

    # Randomize the exact points a bit
    mids = 0.5 * (samples[..., 1:] + samples[..., :-1])
    upper = torch.cat([mids, samples[..., -1:]], -1)
    lower = torch.cat([samples[..., :1], mids], -1)
    # stratified samples in those intervals
    t_rand = torch.rand(samples.shape, device=lower.device)
    samples = lower + (upper - lower) * t_rand
    sampled_points = (
        sources[:, :, None, :] + samples[:, :, :, None] * rays[:, :, None, :]
    )

    # compute the weights for the integral calculation
    weights = samples[..., 1:] - samples[..., :-1]  # [B, R, P]
    # make the last weight to 0
    weights = torch.cat(
        [
            weights,
            torch.Tensor([1e-10]).expand(weights[..., :1].shape).to(weights.device),
        ],
        -1,
    )

    # normalize points between 0 and 1 inside the reference bounding box
    coords = 2 * sampled_points / extent[None, :]

    return coords, target_pixels, weights, patient_idx


def _add_photon_noise(projections, photon_count, torch_rng):
    noisy_data = torch.distributions.Normal(
        torch.exp(-projections) * photon_count,
        torch.sqrt(torch.exp(-projections) * photon_count),
    ).rsample()
    noisy_data = torch.clamp(noisy_data, min=1.0) / photon_count
    projections = -torch.log(noisy_data)

    return projections


def add_photon_noise(projections, photon_count, torch_rng):
    if photon_count is not None:
        if isinstance(projections, list):
            for i in range(len(projections)):
                projections[i] = _add_photon_noise(
                    torch.tensor(projections[i]), photon_count, torch_rng
                ).numpy()
        elif isinstance(projections, torch.Tensor):
            projections = _add_photon_noise(projections, photon_count, torch_rng)
    else:
        pass
    return projections


def train(
    model,
    optimizer,
    train_loader,
    device,
    logger,
    scheduler,
    start_time,
    max_time,
    pbar,
    early_stopping,
    logging_time,
    max_iter,
):
    model.train()

    dset = train_loader.dataset
    extent = (dset.rendering_bbox[1] - dset.rendering_bbox[0]).to(device)
    # training loop dataloader with progress bar using tqdm
    for i, batch in enumerate(train_loader):
        # Get batch of projections
        (
            coords,
            target_pixels,
            weights,
            patient_idx,
        ) = process_batch(batch, dset, device, extent)

        # Get projection data
        out = model(coords)
        # Sum over height to compute projection
        y_pred = (out * weights[:, :, :, None]).sum(dim=2)
        # compute the loss and update the model
        loss = torch.nn.functional.mse_loss(y_pred[:, :, 0], target_pixels)

        # gradient descent step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update the learning rate if necessary
        scheduler.step()

        logging_time_start = time.time()
        logger.train_step(model, loss)
        logging_time += time.time() - logging_time_start
        logger.start_time += time.time() - logging_time_start

        early_stopping.step(logger.last_psnr)
        # elapsed time in seconds
        elapsed_time = time.time() - start_time
        if (
            (elapsed_time > max_time)
            or early_stopping.early_stop
            or logger.train_step_number >= max_iter
        ):
            logger.log_metrics(model, loss, store_volume=True)
            logger.log_volume(model, loss)
            if elapsed_time > max_time:
                log.info("Time limit reached. Finishing training.")
            elif early_stopping.early_stop:
                log.info("Early stopping. Finishing training.")
            return True

        pbar.update(1)

    return False


if __name__ == "__main__":
    main()
