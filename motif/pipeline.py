"""End-to-end MOTIF inference: video -> two temporal embeddings -> POI score."""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from motif import __version__
from motif.extractor import CoeffExtractor
from motif.model import load_encoder
from motif.scoring import BranchReference
from motif.transforms import GlobalTransform, MouthTransform, fold_mouth_projection
from motif.video import extract_video
from motif import weights


@dataclass
class MotifReferenceSet:
    global_videos: list[np.ndarray]
    mouth_videos: list[np.ndarray]
    global_reference: BranchReference
    mouth_reference: BranchReference

    @classmethod
    def enroll(cls, global_videos, mouth_videos):
        if len(global_videos) != len(mouth_videos):
            raise ValueError("global and mouth reference lists differ")
        return cls(
            global_videos=global_videos,
            mouth_videos=mouth_videos,
            global_reference=BranchReference.enroll(global_videos),
            mouth_reference=BranchReference.enroll(mouth_videos),
        )

    def save(self, path):
        payload = {
            "metadata": np.asarray(json.dumps({"format": 1, "motif": __version__})),
            "count": np.asarray(len(self.global_videos), dtype=np.int64),
        }
        for index, embeddings in enumerate(self.global_videos):
            payload[f"global_{index}"] = np.asarray(embeddings, dtype=np.float32)
        for index, embeddings in enumerate(self.mouth_videos):
            payload[f"mouth_{index}"] = np.asarray(embeddings, dtype=np.float32)
        np.savez_compressed(path, **payload)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as payload:
            metadata = json.loads(str(payload["metadata"]))
            if metadata.get("format") != 1:
                raise ValueError("unsupported MOTIF reference-set format")
            count = int(payload["count"])
            global_videos = [payload[f"global_{index}"].copy() for index in range(count)]
            mouth_videos = [payload[f"mouth_{index}"].copy() for index in range(count)]
        return cls.enroll(global_videos, mouth_videos)


class MotifPipeline:
    def __init__(self, device="auto", weights_dir=None, batch_size=32):
        if not torch.cuda.is_available():
            raise RuntimeError("MOTIF requires an NVIDIA GPU for face extraction")
        self.device = "cuda:0" if device == "auto" else device
        self.batch_size = batch_size

        global_path = weights.get_motif_file(weights.GLOBAL_FILE, weights_dir)
        mouth_path = weights.get_motif_file(weights.MOUTH_FILE, weights_dir)
        global_stats = weights.get_motif_file(weights.GLOBAL_STATS_FILE, weights_dir)
        mouth_stats = weights.get_motif_file(weights.MOUTH_STATS_FILE, weights_dir)

        self.global_model, global_args = load_encoder(global_path, self.device)
        self.mouth_model, mouth_args = load_encoder(mouth_path, self.device)
        if global_args["window"] != mouth_args["window"]:
            raise ValueError("global and mouth checkpoints use different window lengths")
        if global_args["input_mode"] != "all" or mouth_args["input_mode"] != "mouthverts":
            raise ValueError("unexpected MOTIF checkpoint input modes")
        fold_mouth_projection(self.mouth_model, mouth_stats)

        self.global_transform = GlobalTransform(global_stats)
        self.mouth_transform = MouthTransform()
        self.extractor = CoeffExtractor(device=self.device, weights_dir=weights_dir)

    @torch.inference_mode()
    def _encode(self, model, windows):
        features = []
        for start in range(0, len(windows), self.batch_size):
            batch = torch.from_numpy(windows[start : start + self.batch_size]).to(self.device)
            features.append(F.normalize(model(batch), p=2, dim=1).cpu())
        return torch.cat(features).numpy()

    def extract_video_embeddings(self, video_path):
        coefficients = extract_video(self.extractor, video_path, self.batch_size)
        global_windows = self.global_transform.windows(coefficients)
        mouth_windows = self.mouth_transform.windows(coefficients)
        return (
            self._encode(self.global_model, global_windows),
            self._encode(self.mouth_model, mouth_windows),
        )

    def extract_reference_set(self, reference_videos):
        paths = [Path(path) for path in reference_videos]
        if len(paths) < 2:
            raise ValueError("MOTIF requires at least two reference videos")
        global_videos, mouth_videos = [], []
        for path in paths:
            global_embeddings, mouth_embeddings = self.extract_video_embeddings(path)
            global_videos.append(global_embeddings)
            mouth_videos.append(mouth_embeddings)
        return MotifReferenceSet.enroll(global_videos, mouth_videos)

    def score(self, reference_set, test_video):
        if not isinstance(reference_set, MotifReferenceSet):
            reference_set = MotifReferenceSet.load(reference_set)
        global_embeddings, mouth_embeddings = self.extract_video_embeddings(test_video)
        global_score = reference_set.global_reference.calibrated_score(global_embeddings)
        mouth_score = reference_set.mouth_reference.calibrated_score(mouth_embeddings)
        return {
            "score": float(0.5 * (global_score + mouth_score)),
            "global_score": float(global_score),
            "mouth_score": float(mouth_score),
        }
