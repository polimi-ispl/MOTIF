"""Resolve MOTIF and upstream 3DDFA-V3 inference assets."""

import os
from pathlib import Path

MOTIF_HF_REPO = "heyGio/MOTIF"
TDDFA_HF_REPO = "Zidu-Wang/3DDFA-V3"

GLOBAL_FILE = "motif_global.pth"
MOUTH_FILE = "motif_mouth.pth"
GLOBAL_STATS_FILE = "norm_stats_all.npz"
MOUTH_STATS_FILE = "norm_stats_mvrecon.npz"
RETINAFACE_FILE = "retinaface_resnet50_2020-07-20_old_torch.pth"
LANDMARK_FILE = "large_base_net.pth"
NET_RECON_FILE = "net_recon.pth"


def _local_dir(weights_dir=None):
    return weights_dir or os.environ.get("MOTIF_WEIGHTS_DIR")


def _local_file(filename, directory):
    path = Path(directory) / filename
    if not path.is_file():
        raise FileNotFoundError(
            f"Weight file '{filename}' not found in '{directory}'. Place every "
            "required file there, or unset --weights-dir/MOTIF_WEIGHTS_DIR to "
            "download from the Hugging Face Hub."
        )
    return str(path)


def _download(repo_id, filename, repo_type="model"):
    from huggingface_hub import hf_hub_download

    try:
        return hf_hub_download(repo_id=repo_id, repo_type=repo_type, filename=filename)
    except Exception as error:
        raise RuntimeError(
            f"Could not download '{filename}' from https://huggingface.co/{repo_id}. "
            "The MOTIF weights repository is private during review, so authenticate "
            "with a Hugging Face token or provide --weights-dir."
        ) from error


def get_motif_file(filename, weights_dir=None):
    directory = _local_dir(weights_dir)
    return _local_file(filename, directory) if directory else _download(MOTIF_HF_REPO, filename)


def get_tddfa_asset(filename, weights_dir=None):
    directory = _local_dir(weights_dir)
    if directory:
        return _local_file(filename, directory)
    return _download(TDDFA_HF_REPO, f"assets/{filename}", repo_type="dataset")
