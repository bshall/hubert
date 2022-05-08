# HuBERT

The HuBERT-Soft and HuBERT-Discrete models for [soft-vc](https://github.com/bshall/soft-vc).

Relevant links:
- [Official HuBERT repo](https://github.com/pytorch/fairseq)
- [HuBERT paper](https://arxiv.org/abs/2106.07447)
- [Soft-VC repo](https://github.com/bshall/soft-vc)
- [Soft-VC paper]()

## Example Usage

### Soft Speech Units

```python
import torch, torchaudio

# Load checkpoint
hubert = torch.hub.load("bshall/hubert:main", "hubert_soft").cuda()

# Load audio
wav, sr = torchaudio.load("path/to/wav")
assert sr == 16000
wav = wav.unsqueeze(0).cuda()

# Extract soft speech units
with torch.no_grad():
    x, _ = hubert.encode(wav)
    units = hubert.proj(x)
```

### Discrete Speech Units

```python
import torch, torchaudio

# Load checkpoints
hubert = torch.hub.load("bshall/hubert:main", "hubert_discrete").cuda()
kmeans = torch.hub.load("bshall/hubert:main", "kmeans100")

# Load audio
wav, sr = torchaudio.load("path/to/wav")
assert sr == 16000
wav = wav.unsqueeze(0).cuda()

# Extract hubert features from the 7th transformer layer
with torch.no_grad():
    x, _ = hubert.encode(wav, layer=7)

# Extract discrete speech units
units = kmeans.predict(x.squeeze().cpu().numpy())
```

## Training

**Step 1**: Download and extract LibriSpeech

**Step 2**: Encode LibriSpeech using the HuBERT-Discrete model and `encode.py` script (setting `--layer=7`):

```
usage: encode.py [-h] [--extension EXTENSION] [--model {hubert_soft,hubert_discrete}] [--layer LAYER] in-dir out-dir

Encode an audio dataset.

positional arguments:
  in-dir                path to the dataset directory.
  out-dir               path to the output directory.

optional arguments:
  -h, --help            show this help message and exit
  --extension EXTENSION
                        extension of the audio files.
  --model {hubert_soft,hubert_discrete}
                        available models
  --layer LAYER         the selected transformer layer (defaults to the last layer)
```

**Step 3**: Discretize the extracted features using the k-means checkpoint and `discretize.py` script:

```
usage: discretize.py [-h] in-dir out-dir

Discretize HuBERT features.

positional arguments:
  in-dir      path to the dataset directory.
  out-dir     path to the output directory.

optional arguments:
  -h, --help  show this help message and exit
```

**Step 5**: Train the HuBERT-Soft model using the `train.py` script:

```
usage: train.py [-h] [--resume RESUME] [--warmstart] [--mask] [--alpha ALPHA] dataset-dir checkpoint-dir

Train HuBERT soft content encoder.

positional arguments:
  dataset-dir      path to the data directory.
  checkpoint-dir   path to the checkpoint directory.

optional arguments:
  -h, --help       show this help message and exit
  --resume RESUME  path to the checkpoint to resume from.
  --warmstart      whether to initialize from the fairseq HuBERT checkpoint.
  --mask           whether to use input masking.
  --alpha ALPHA    weight for the masked loss.
```