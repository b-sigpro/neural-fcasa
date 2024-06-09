from __future__ import annotations

from pathlib import Path

import numpy as np

import torch

from aiaccel.torch.datasets import CachedDataset, HDF5Dataset


class HDF5WavActDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_path: Path | str,
        duration: int | None = None,
        sr: int | None = None,
        hop_length: int | None = None,
        randperm_mic: bool = True,
        randperm_spk: bool = True,
    ) -> None:
        super().__init__()

        self._dataset = CachedDataset(HDF5Dataset(dataset_path))

        self.duration = duration
        self.sr = sr
        self.hop_length = hop_length

        self.randperm_mic = randperm_mic
        self.randperm_spk = randperm_spk

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int):
        item = self._dataset[index]
        wav = item["wav"]
        act = item["act"]

        if self.duration is not None:
            duration = self.sr * self.duration // self.hop_length
            t_start_act = np.random.randint(0, act.shape[1] - duration + 1)
            t_end_act = t_start_act + duration

            act = act[:, t_start_act:t_end_act]

            t_start = self.hop_length * t_start_act
            t_end = self.hop_length * t_end_act
            wav = wav[:, t_start:t_end]

        if self.randperm_mic:
            wav = wav[torch.randperm(wav.shape[0])]

        if self.randperm_spk:
            act = act[torch.randperm(act.shape[0])]

        return wav, act
