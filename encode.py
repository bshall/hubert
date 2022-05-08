import argparse
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torchaudio
from torchaudio.functional import resample

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def encode_dataset(args):
    print(f"Loading hubert checkpoint")
    hubert = torch.hub.load("bshall/hubert:main", args.model).cuda()

    print(f"Encoding dataset at {args.in_dir}")
    for in_path in tqdm(list(args.in_dir.rglob(f"*{args.extension}"))):
        wav, sr = torchaudio.load(in_path)
        wav = resample(wav, sr, 16000)
        wav = wav.unsqueeze(0).cuda()
        wav = F.pad(wav, ((400 - 320) // 2, (400 - 320) // 2))

        # Extract hubert features from the args.layer transformer layer
        with torch.inference_mode():
            x, _ = hubert.encode(wav, layer=args.layer)
            if args.layer is None:
                x = hubert.proj(x)

        out_path = args.out_dir / in_path.relative_to(args.in_dir)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path.with_suffix(".npy"), x.squeeze(0).cpu().numpy())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encode an audio dataset.")
    parser.add_argument(
        "in_dir",
        metavar="in-dir",
        help="path to the dataset directory.",
        type=Path,
    )
    parser.add_argument(
        "out_dir",
        metavar="out-dir",
        help="path to the output directory.",
        type=Path,
    )
    parser.add_argument(
        "--extension",
        help="extension of the audio files.",
        default=".flac",
        type=str,
    )
    parser.add_argument(
        "--model",
        help="available models",
        choices=["hubert_soft", "hubert_discrete"],
        default="hubert_soft",
    )
    parser.add_argument(
        "--layer",
        help="the transformer layer to extract from",
        default=None,
        type=int,
    )
    args = parser.parse_args()
    encode_dataset(args)