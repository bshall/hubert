# HuBERT

Training and inference scripts for the HuBERT content encoders in [A Comparison of Discrete and Soft Speech Units for Improved Voice Conversion](https://ieeexplore.ieee.org/abstract/document/9746484). 
For more details see [soft-vc](https://github.com/bshall/).

<div align="center">
    <img width="100%" alt="Soft-VC"
      src="https://raw.githubusercontent.com/bshall/hubert/main/diagram.png">
</div>
<div>
  <sup>
    <strong>Fig 1:</strong> Architecture of the voice conversion system. a) The <strong>discrete</strong> content encoder clusters audio features to produce a sequence of discrete speech units. b) The <strong>soft</strong> content encoder is trained to predict the discrete units. The acoustic model transforms the discrete/soft speech units into a target spectrogram. The vocoder converts the spectrogram into an audio waveform.
  </sup>
</div>

[TOC]

Relevant links:
- [Soft-VC repo](https://github.com/bshall/soft-vc)
- [Soft-VC paper](https://ieeexplore.ieee.org/abstract/document/9746484)
- [Official HuBERT repo](https://github.com/pytorch/fairseq)
- [HuBERT paper](https://arxiv.org/abs/2106.07447)

## Example Usage

### Extract Speech Units

```python
import torch, torchaudio

# Load checkpoint (either hubert_soft or hubert_discrete)
hubert = torch.hub.load("bshall/hubert:main", "hubert_soft").cuda()

# Load audio
wav, sr = torchaudio.load("path/to/wav")
assert sr == 16000
wav = wav.unsqueeze(0).cuda()

# Extract speech units
units = hubert.units(x)
```

## Training

**Step 1**: Download and extract the [LibriSpeech](https://www.openslr.org/12) corpus.

**Step 2**: Encode LibriSpeech using the HuBERT-Discrete model and `encode.py` script:

```
usage: encode.py [-h] [--extension EXTENSION] [--model {hubert_soft,hubert_discrete}] in-dir out-dir

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
```

for example:

```
python encode.py path/to/LibriSpeech/wavs path/to/LibriSpeech/units --model hubert_discrete
```

**Step 3**: Train the HuBERT-Soft model using the `train.py` script:

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