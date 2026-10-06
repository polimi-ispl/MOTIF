"""Decode every frame of a video and extract its 3DMM coefficients."""

import av
import numpy as np
import torch


def extract_video(extractor, path, batch_size=32):
    parts = []
    buffer = []
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            rgb = np.ascontiguousarray(frame.to_ndarray(format="rgb24"))
            buffer.append(torch.from_numpy(rgb).permute(2, 0, 1))
            if len(buffer) == batch_size:
                parts.append(extractor.process_frames(torch.stack(buffer)))
                buffer = []
    if buffer:
        parts.append(extractor.process_frames(torch.stack(buffer)))
    if not parts:
        raise ValueError(f"no decodable frames in '{path}'")

    output = {key: torch.cat([part[key] for part in parts]).numpy() for key in parts[0]}
    return {
        "id": output["id"].astype(np.float32),
        "exp": output["exp"].astype(np.float32),
        "angle": output["angle"].astype(np.float32),
        "trans": output["trans"].astype(np.float32),
        "valid": output["valid"].astype(bool),
    }
