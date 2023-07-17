import pdb

from torch.utils.data import DataLoader

from dataset.sampler import PerPatientReconBatchSampler

from dataset.cbct_fast_recon_dataset import MultiVolumeConebeamReconDataset
from dataset.cbct_dataset import MultiVolumeConebeamDataset

from pathlib import Path


def get_dataset(dataset_cfg, stage):
    if dataset_cfg.name == "reconstructions":
        return MultiVolumeConebeamReconDataset(
            volumes_dir=dataset_cfg.path,
            original_volumes_dir=dataset_cfg.original_volumes_path,
            num_vols=dataset_cfg.num_vols,
            stage=stage,
        )
    elif dataset_cfg.name == "projections":
        return MultiVolumeConebeamDataset(
            volumes_dir=dataset_cfg.path,
            num_vols=dataset_cfg.num_vols,
            num_projs=dataset_cfg.num_projs,
            num_steps=dataset_cfg.num_steps,
            projs_sample_step=dataset_cfg.projs_sample_step,
            stage=dataset_cfg.stage,
            noisy_projections=dataset_cfg.noisy_projections,
            volume_id=dataset_cfg.volume_id,
        )
    else:
        raise ValueError("Unknown dataset name: {}".format(dataset_cfg.name))


def get_dataloader(dataset_cfg, stage_cfg, stage):
    dataset = get_dataset(dataset_cfg, stage)
    if stage_cfg.per_patient_batching:
        batch_sampler = PerPatientReconBatchSampler(
            dataset,
            batch_size=stage_cfg.batch_size,
        )
        return DataLoader(
            dataset=dataset,
            batch_sampler=batch_sampler,
            num_workers=stage_cfg.num_workers,
            shuffle=False,
            pin_memory=False,
        )
    else:
        return DataLoader(
            dataset=dataset,
            batch_size=stage_cfg.batch_size,
            shuffle=stage_cfg.shuffle,
            num_workers=stage_cfg.num_workers,
            pin_memory=False,
        )
