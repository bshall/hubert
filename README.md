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
hubert = torch.hub.load("bshall/hubert:main", "hubert").cuda()
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