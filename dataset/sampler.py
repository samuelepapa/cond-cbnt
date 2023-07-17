import pdb

import torch
import numpy as np
from dataset.cbct_recon_dataset import MultiVolumeConebeamReconDataset

import logging


class PerPatientReconBatchSampler(torch.utils.data.Sampler):
    def __init__(self, data_source: MultiVolumeConebeamReconDataset, batch_size):
        super().__init__(data_source)
        self.data_source = data_source
        self.batch_size = batch_size

        self.patient_start_indices = []
        self.pixels_per_patient = []

        for patient in range(self.data_source.num_patients):
            self.pixels_per_patient.append(
                int(np.prod(self.data_source.volume_shapes[patient]))
            )
        # Do this once to get the batched indices
        # self.__iter__()

    def __iter__(self):
        self.batched_indices = []
        self.batched_patient_indices = []
        self.batched_content = []
        for patient in range(self.data_source.num_patients):
            # Get a list of randomly permuted indices for the patient
            indices = torch.randperm(
                self.pixels_per_patient[patient], dtype=torch.int32
            )
            patient_indices = torch.full((len(indices),), patient, dtype=torch.int32)

            rounded_size = len(indices) - (len(indices) % self.batch_size)
            indices = indices[:rounded_size]
            patient_indices = patient_indices[:rounded_size]

            # concat indices and patient_indices along the last dim
            content = torch.stack((indices, patient_indices), dim=-1)
            content = content.reshape(-1, self.batch_size, 2)

            self.batched_content.append(content)

        self.batched_content = torch.cat(self.batched_content, dim=0)
        # random permute along dimension 0
        self.batched_content = self.batched_content[
            torch.randperm(self.batched_content.shape[0])
        ].numpy()
        return iter(self.batched_content)

    def __len__(self):
        return np.sum([i // self.batch_size for i in self.pixels_per_patient])
