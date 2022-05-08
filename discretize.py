import argparse
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def discretize_dataset(args):
    logger.info("Loading k-means checkpoint")
    kmeans = torch.hub.load("bshall/hubert:main", "kmeans100")

    logger.info(f"Discretizing dataset at {args.in_dir}")
    for in_path in tqdm(list(args.in_dir.rglob("*.npy"))):
        x = np.load(in_path)
        x = kmeans.predict(x)

        out_path = args.out_dir / in_path.relative_to(args.in_dir)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path.with_suffix(".npy"), x)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Discretize an audio dataset.")
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
    args = parser.parse_args()
    discretize_dataset(args)
