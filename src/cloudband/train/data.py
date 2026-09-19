"""Fastai DataLoaders built from CloudSEN12+ samples."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import torch
from fastai.data.core import DataLoaders
from torch.utils.data import Dataset

from cloudband.datasets import cloudsen12 as dataset
from cloudband.pipelines.phase0 import select_rgn
from cloudband.train.normalize import dynamic_z_score

SampleReader = Callable[[pd.DataFrame, int, bool], dataset.Sample]


class CloudSen12Dataset(Dataset):
    """Indexes into a CloudSEN12+ table, returning RGN image and annotation pairs."""

    def __init__(
        self,
        table: pd.DataFrame,
        crop_to_valid: bool = True,
        read_sample: SampleReader = dataset.read_sample,
    ):
        self.table = table
        self.crop_to_valid = crop_to_valid
        self.read_sample = read_sample

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.read_sample(self.table, index, self.crop_to_valid)
        image = dynamic_z_score(select_rgn(sample.image).astype(np.float32))
        annotation = sample.annotation.astype(np.int64)
        return torch.from_numpy(image), torch.from_numpy(annotation)


def build_dataloaders(
    train_table: pd.DataFrame,
    valid_table: pd.DataFrame,
    micro_batch_size: int,
    num_workers: int = 0,
    read_sample: SampleReader = dataset.read_sample,
) -> DataLoaders:
    """Wrap train and validation tables into fastai DataLoaders.

    micro_batch_size is the per-step batch the hardware can hold, independent
    of the protocol's effective batch size; gradient accumulation in the
    training loop makes up the difference.
    """
    train_dataset = CloudSen12Dataset(train_table, read_sample=read_sample)
    valid_dataset = CloudSen12Dataset(valid_table, read_sample=read_sample)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=micro_batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=micro_batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return DataLoaders(train_loader, valid_loader)