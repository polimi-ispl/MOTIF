"""Build the two paper inputs from per-frame 3DMM coefficients."""

import numpy as np
import torch
from torch import nn

WINDOW = 75
STRIDE = WINDOW // 2


def valid_windows(coefficients, window=WINDOW, stride=STRIDE):
    valid = np.asarray(coefficients["valid"], dtype=bool)
    runs = []
    start = None
    for index, value in enumerate(valid):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, len(valid)))

    long_runs = [(start, end) for start, end in runs if end - start >= window]
    if long_runs:
        spans = []
        for start, end in long_runs:
            starts = list(range(start, end - window + 1, stride))
            if starts[-1] != end - window:
                starts.append(end - window)
            spans.extend((offset, window) for offset in starts)
        return spans

    if not runs:
        raise ValueError("video has no frames with a detected face")
    start, end = max(runs, key=lambda run: run[1] - run[0])
    return [(start, end - start)]


def _window_field(coefficients, name, start, length, window=WINDOW):
    values = np.asarray(coefficients[name], dtype=np.float32)[start : start + length]
    if length < window:
        values = np.concatenate(
            [values, np.repeat(values[-1:], window - length, axis=0)], axis=0
        )
    return values


class GlobalTransform:
    def __init__(self, stats_path):
        with np.load(stats_path) as stats:
            self.mean = stats["mean"].astype(np.float32)
            self.std = np.maximum(stats["std"].astype(np.float32), 1e-6)

    def windows(self, coefficients):
        output = []
        for start, length in valid_windows(coefficients):
            sequence = np.concatenate(
                [
                    _window_field(coefficients, "id", start, length),
                    _window_field(coefficients, "exp", start, length),
                    _window_field(coefficients, "angle", start, length),
                    _window_field(coefficients, "trans", start, length),
                ],
                axis=1,
            )
            output.append((sequence - self.mean) / self.std)
        return np.stack(output).astype(np.float32)


class MouthTransform:
    def windows(self, coefficients):
        output = []
        for start, length in valid_windows(coefficients):
            sequence = np.concatenate(
                [
                    _window_field(coefficients, "id", start, length),
                    _window_field(coefficients, "exp", start, length),
                ],
                axis=1,
            )
            output.append(sequence - sequence.mean(axis=0, keepdims=True))
        return np.stack(output).astype(np.float32)


def fold_mouth_projection(model, stats_path):
    """Fold lip-vertex reconstruction and standardization into the tokenizer."""
    with np.load(stats_path) as stats:
        std = np.maximum(stats["std"].astype(np.float64), 1e-6)
        identity_basis = stats["Bid"].astype(np.float64)
        expression_basis = stats["Bexp"].astype(np.float64)

    embed = model.embed
    weights = embed.weight.detach().cpu().numpy().astype(np.float64) / std
    folded_weights = np.concatenate(
        [weights @ identity_basis, weights @ expression_basis], axis=1
    )
    folded = nn.Linear(
        144, embed.out_features, bias=embed.bias is not None,
        device=embed.weight.device, dtype=embed.weight.dtype,
    )
    folded.weight.data.copy_(
        torch.from_numpy(folded_weights).to(embed.weight.device, embed.weight.dtype)
    )
    if embed.bias is not None:
        folded.bias.data.copy_(embed.bias.detach())
    model.embed = folded
    return model
