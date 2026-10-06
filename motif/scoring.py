"""Per-POI whitening, leave-one-reference-out calibration, and score fusion."""

from dataclasses import dataclass

import numpy as np
from sklearn.covariance import LedoitWolf

EPS = 1e-8


def _normalize(values):
    values = np.asarray(values, dtype=np.float64)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def _covariance(values):
    try:
        covariance = LedoitWolf(assume_centered=False).fit(values).covariance_
    except Exception:
        variance = np.maximum(np.var(values, axis=0), EPS)
        covariance = np.diag(variance)
    return (covariance + covariance.T) / 2.0


def _inverse_sqrt(covariance):
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    floor = max(EPS, 1e-8 * float(eigenvalues.max(initial=0.0)))
    eigenvalues = np.maximum(eigenvalues, floor)
    result = (eigenvectors * eigenvalues ** -0.5) @ eigenvectors.T
    return (result + result.T) / 2.0


@dataclass
class PoiWhiten:
    matrix: np.ndarray
    bank: np.ndarray

    @classmethod
    def fit(cls, bank):
        bank = np.asarray(bank, dtype=np.float64)
        if bank.ndim != 2 or len(bank) < 2 or not np.isfinite(bank).all():
            raise ValueError("reference bank needs at least two finite embeddings")
        scale = max(float(np.abs(bank).max(initial=0.0)), 1.0)
        if float(bank.std(axis=0).max(initial=0.0)) <= 1e-12 * scale:
            raise ValueError("reference bank contains no measurable variation")
        matrix = _inverse_sqrt(_covariance(bank))
        return cls(matrix=matrix, bank=_normalize(bank @ matrix))

    def score(self, query):
        query = _normalize(np.asarray(query, dtype=np.float64) @ self.matrix)
        return float((self.bank @ query.T).max())


@dataclass
class BranchReference:
    readout: PoiWhiten
    mean: float
    std: float

    @classmethod
    def enroll(cls, video_embeddings):
        if len(video_embeddings) < 2:
            raise ValueError("MOTIF requires at least two usable reference videos")
        loo_scores = []
        for index, query in enumerate(video_embeddings):
            bank = np.concatenate(
                [value for other, value in enumerate(video_embeddings) if other != index]
            )
            loo_scores.append(PoiWhiten.fit(bank).score(query))
        deviation = float(np.std(loo_scores, ddof=1))
        if not np.isfinite(deviation) or deviation < 1e-6:
            deviation = 1e-6
        return cls(
            readout=PoiWhiten.fit(np.concatenate(video_embeddings)),
            mean=float(np.mean(loo_scores)),
            std=deviation,
        )

    def calibrated_score(self, embeddings):
        return (self.readout.score(embeddings) - self.mean) / self.std
