from typing import Any

from pathlib import Path

from torch.utils.data import DataLoader

import lightning as lt

from aiaccel.torch.datasets import scatter_dataset

from neural_fcasa.datasets.hdf5_wavact_dataset import HDF5WavActDataset


class DataModule(lt.LightningDataModule):
    def __init__(
        self,
        train_dataset_path: str | Path,
        val_dataset_path: str | Path,
        batch_size: int,
        duration: int | None = None,
        sr: int | None = None,
        hop_length: int | None = None,
        randperm_mic: bool = True,
        randperm_spk: bool = True,
        num_workers: int = 10,
    ):
        super().__init__()

        self.train_dataset_path = train_dataset_path
        self.val_dataset_path = val_dataset_path

        self.default_dataloader_kwargs: dict[str, Any] = dict(
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=True,
            shuffle=True,
        )

        self.default_dataset_kwargs: dict[str, Any] = dict(
            duration=duration,
            sr=sr,
            hop_length=hop_length,
            randperm_mic=randperm_mic,
            randperm_spk=randperm_spk,
        )

    def setup(self, stage: str | None):
        if stage == "fit":
            self.train_dataset = scatter_dataset(
                HDF5WavActDataset(self.train_dataset_path, **self.default_dataset_kwargs)
            )
            self.val_dataset = scatter_dataset(HDF5WavActDataset(self.val_dataset_path, **self.default_dataset_kwargs))

            print(f"Dataset size: {len(self.train_dataset)=},  {len(self.val_dataset)=}")
        else:
            raise ValueError("`stage` is not 'fit'.")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            drop_last=True,
            **self.default_dataloader_kwargs,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            drop_last=False,
            **self.default_dataloader_kwargs,
        )
