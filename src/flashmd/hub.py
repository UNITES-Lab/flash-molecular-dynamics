"""HuggingFace Hub integration for loading pre-trained FlashMD models."""

import torch
from pathlib import Path
from typing import Optional


def from_pretrained(
    repo_id: str = "pingzhili/cg-schnet",
    filename: str = "model_and_prior.pt",
    cache_dir: Optional[str] = None,
    revision: Optional[str] = None,
) -> torch.nn.Module:
    """Download and load a pre-trained model from HuggingFace Hub.

    Parameters
    ----------
    repo_id : str
        HuggingFace repository ID (default: "pingzhili/cg-schnet")
    filename : str
        Name of the model file to download (default: "model_and_prior.pt")
    cache_dir : str, optional
        Directory to cache downloaded files
    revision : str, optional
        Git revision (branch, tag, or commit hash)

    Returns
    -------
    torch.nn.Module
        Loaded model checkpoint
    """
    from huggingface_hub import hf_hub_download

    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        cache_dir=cache_dir,
        revision=revision,
    )

    from flashmd.models.pyg_forward_compatibility import (
        load_and_adapt_old_checkpoint,
    )

    model = load_and_adapt_old_checkpoint(local_path)
    return model


def download_file(
    repo_id: str = "pingzhili/cg-schnet",
    filename: str = "1enh_configurations.pt",
    cache_dir: Optional[str] = None,
    revision: Optional[str] = None,
) -> Path:
    """Download a file from HuggingFace Hub and return the local path.

    Useful for downloading initial configurations, data files, etc.

    Parameters
    ----------
    repo_id : str
        HuggingFace repository ID (default: "pingzhili/cg-schnet")
    filename : str
        Name of the file to download
    cache_dir : str, optional
        Directory to cache downloaded files
    revision : str, optional
        Git revision (branch, tag, or commit hash)

    Returns
    -------
    Path
        Local path to the downloaded file
    """
    from huggingface_hub import hf_hub_download

    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        cache_dir=cache_dir,
        revision=revision,
    )
    return Path(local_path)
