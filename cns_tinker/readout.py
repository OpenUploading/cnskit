"""Local ridge readout fitting. Does not update connectome weights or dynamics."""
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np


def _names(values):
    values = tuple(values)
    if not values or any(not isinstance(v, str) or not v for v in values) or len(set(values)) != len(values):
        raise ValueError("Names must be nonempty, unique strings")
    return values


def _matrix(values, columns):
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 2 or result.shape[0] == 0 or result.shape[1] != columns or not np.isfinite(result).all():
        raise ValueError("Expected a finite, nonempty sample-by-channel matrix")
    return result


@dataclass(frozen=True)
class RidgeReadout:
    feature_names: tuple[str, ...]
    output_names: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    bias: np.ndarray
    alpha: float

    @classmethod
    def fit(cls, activity, targets, *, feature_names, output_names, alpha=1.0):
        """Fit ONLY on training episodes; evaluate unseen episodes separately."""
        features, outputs = _names(feature_names), _names(output_names)
        x, y = _matrix(activity, len(features)), _matrix(targets, len(outputs))
        if len(x) != len(y) or len(x) < 2:
            raise ValueError("Training needs at least two aligned samples")
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be finite and positive")
        mean, scale = x.mean(axis=0), x.std(axis=0)
        scale = np.where(scale > 1e-12, scale, 1.0)
        z, bias = (x - mean) / scale, y.mean(axis=0)
        centered = y - bias
        # Solve the smaller system; callers should select a bounded readout population.
        if z.shape[1] <= z.shape[0]:
            weights = np.linalg.solve(z.T @ z + alpha * np.eye(z.shape[1]), z.T @ centered)
        else:
            weights = z.T @ np.linalg.solve(z @ z.T + alpha * np.eye(z.shape[0]), centered)
        if not np.isfinite(weights).all():
            raise ValueError("Nonfinite fit; rescale input values")
        return cls(features, outputs, mean, scale, weights, bias, float(alpha))

    def predict(self, activity, *, feature_names):
        if tuple(feature_names) != self.feature_names:
            raise ValueError("Feature names and order must match the fitted checkpoint")
        x = _matrix(activity, len(self.feature_names))
        return ((x - self.mean) / self.scale) @ self.weights + self.bias

    def evaluate(self, activity, targets, *, feature_names):
        predicted = self.predict(activity, feature_names=feature_names)
        y = _matrix(targets, len(self.output_names))
        if y.shape != predicted.shape:
            raise ValueError("Evaluation targets must align with predictions")
        return {name: float(value) for name, value in zip(self.output_names, np.mean((predicted-y)**2, axis=0))}

    def save(self, path):
        payload = {"schema": "cnskit.ridge.v1", "feature_names": self.feature_names,
                   "output_names": self.output_names, "alpha": self.alpha,
                   "mean": self.mean.tolist(), "scale": self.scale.tolist(),
                   "weights": self.weights.tolist(), "bias": self.bias.tolist(),
                   "training_scope": "readout_only; connectome unchanged"}
        # Explicitly refuse overwrite; checkpoints are caller-owned artifacts.
        with Path(path).open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, allow_nan=False, indent=2)

    @classmethod
    def load(cls, path):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema") != "cnskit.ridge.v1":
            raise ValueError("Unsupported readout checkpoint schema")
        features, outputs = _names(payload["feature_names"]), _names(payload["output_names"])
        arrays = [np.asarray(payload[k], dtype=float) for k in ("mean", "scale", "weights", "bias")]
        shapes = [(len(features),), (len(features),), (len(features),len(outputs)), (len(outputs),)]
        if any(a.shape != shape or not np.isfinite(a).all() for a,shape in zip(arrays,shapes)):
            raise ValueError("Invalid checkpoint dimensions or values")
        alpha = float(payload["alpha"])
        if np.any(arrays[1] <= 0) or not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("Invalid checkpoint scale or regularization")
        return cls(features, outputs, *arrays, alpha)
