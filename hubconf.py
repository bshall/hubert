dependencies = ["torch", "torchaudio", "sklearn"]

URLS = {
    "hubert-discrete": "https://github.com/bshall/hubert/releases/download/v0.2/hubert-discrete-96b248c5.pt",
    "hubert-soft": "https://github.com/bshall/hubert/releases/download/v0.2/hubert-soft-35d9f29f.pt",
    "kmeans100": "https://github.com/bshall/hubert/releases/download/v0.2/kmeans100-50f36a95.pt",
}

import torch
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

from sklearn.cluster import KMeans

from hubert import HubertDiscrete, HubertSoft


def hubert_discrete(
    pretrained: bool = True,
    progress: bool = True,
) -> HubertDiscrete:
    r"""HuBERT-Discrete from `"A Comparison of Discrete and Soft Speech Units for Improved Voice Conversion"`.
    Args:
        pretrained (bool): load pretrained weights into the model
        progress (bool): show progress bar when downloading model
    """
    kmeans = kmeans100(pretrained=pretrained, progress=progress)
    hubert = HubertDiscrete(kmeans)
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            URLS["hubert-discrete"], progress=progress
        )
        consume_prefix_in_state_dict_if_present(checkpoint["hubert"], "module.")
        hubert.load_state_dict(checkpoint["hubert"])
        hubert.eval()
    return hubert


def hubert_soft(
    pretrained: bool = True,
    progress: bool = True,
) -> HubertSoft:
    r"""HuBERT-Soft from `"A Comparison of Discrete and Soft Speech Units for Improved Voice Conversion"`.
    Args:
        pretrained (bool): load pretrained weights into the model.
        progress (bool): show progress bar when downloading model.
    """
    hubert = HubertSoft()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            URLS["hubert-soft"],
            progress=progress,
        )
        consume_prefix_in_state_dict_if_present(checkpoint["hubert"], "module.")
        hubert.load_state_dict(checkpoint["hubert"])
        hubert.eval()
    return hubert


def _kmeans(
    num_clusters: int, pretrained: bool = True, progress: bool = True
) -> KMeans:
    kmeans = KMeans(num_clusters)
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            URLS[f"kmeans{num_clusters}"], progress=progress
        )
        kmeans.__dict__["n_features_in_"] = checkpoint["n_features_in_"]
        kmeans.__dict__["_n_threads"] = checkpoint["_n_threads"]
        kmeans.__dict__["cluster_centers_"] = checkpoint["cluster_centers_"].numpy()
    return kmeans


def kmeans100(pretrained: bool = True, progress: bool = True) -> KMeans:
    r"""
    k-means checkpoint for HuBERT-Discrete with 100 clusters.
    Args:
        pretrained (bool): load pretrained weights into the model
        progress (bool): show progress bar when downloading model
    """
    return _kmeans(100, pretrained, progress)
