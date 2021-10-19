import random
from pathlib import Path
import numpy as np
import json

import torch
from torch.utils.data import Dataset
import torchaudio


class AcousticUnitsDataset(Dataset):
    def __init__(
        self,
        root: Path,
        sample_rate: int = 16000,
        label_rate: int = 50,
        min_samples: int = 32000,
        max_samples: int = 250000,
        train: bool = True,
    ):
        with open(root / "lengths.json") as file:
            self.lenghts = json.load(file)

        pattern = "train-*/**/*.wav" if train else "dev-*/**/*.wav"
        self.metadata = [
            path
            for path in root.rglob(pattern)
            if self.lenghts[path.stem] > min_samples
        ]
        self.sample_rate = sample_rate
        self.label_rate = label_rate
        self.min_samples = min_samples
        self.max_samples = max_samples
        self.train = train

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        path = self.metadata[index]

        codes = np.load(path.with_suffix(".npy"))
        wav, _ = torchaudio.load(path)

        return wav, torch.from_numpy(codes).long()

    def collate(self, batch):
        wavs, codes = zip(*batch)
        wavs, codes = list(wavs), list(codes)

        wav_lengths = [wav.size(-1) for wav in wavs]
        code_lengths = [code.size(-1) for code in codes]

        wav_frames = min(self.max_samples, *wav_lengths)

        collated_wavs, wav_offsets = [], []
        for wav in wavs:
            wav_diff = wav.size(-1) - wav_frames
            wav_offset = random.randint(0, wav_diff)
            wav = wav[:, wav_offset : wav_offset + wav_frames]

            collated_wavs.append(wav)
            wav_offsets.append(wav_offset)

        rate = self.label_rate / self.sample_rate
        code_offsets = [round(wav_offset * rate) for wav_offset in wav_offsets]
        code_frames = round(wav_frames * rate)
        remaining_code_frames = [
            length - offset for length, offset in zip(code_lengths, code_offsets)
        ]
        code_frames = min(code_frames, *remaining_code_frames)

        collated_codes = []
        for code, code_offset in zip(codes, code_offsets):
            code = code[code_offset : code_offset + code_frames]
            collated_codes.append(code)

        wavs = torch.stack(collated_wavs, dim=0)
        codes = torch.stack(collated_codes, dim=0)

        return wavs, codes
