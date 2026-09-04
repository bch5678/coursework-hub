"""Small PyTorch adapter contract for future single- and multimodal models."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import pandas as pd


class ModelAdapter(Protocol):
    """A model-specific adapter only needs evaluation mode and batch logits."""

    def eval(self) -> Any: ...

    def predict_logits(self, batch: dict[str, Any]) -> Any: ...


def _to_device(value: Any, device: Any) -> Any:
    if hasattr(value, "to"):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_device(item, device) for item in value)
    return value


def logits_to_positive_probability(logits: Any) -> Any:
    import torch

    if logits.ndim == 1:
        return torch.sigmoid(logits)
    if logits.ndim == 2 and logits.shape[1] == 1:
        return torch.sigmoid(logits[:, 0])
    if logits.ndim == 2 and logits.shape[1] == 2:
        return torch.softmax(logits, dim=1)[:, 1]
    raise ValueError(f"Expected logits shaped [B], [B,1] or [B,2], got {tuple(logits.shape)}")


def write_torch_predictions(
    adapter: ModelAdapter,
    dataloader: Any,
    output_path: str | Path,
    *,
    device: str = "cpu",
) -> Path:
    import torch

    target = torch.device(device)
    adapter.eval()
    rows = []
    with torch.inference_mode():
        for batch in dataloader:
            sample_ids = list(batch["sample_id"])
            device_batch = _to_device(batch, target)
            probability = logits_to_positive_probability(adapter.predict_logits(device_batch)).detach().cpu()
            if probability.ndim != 1 or len(probability) != len(sample_ids):
                raise ValueError("Adapter output batch size does not match sample_id")
            rows.extend({"sample_id": sample_id, "y_score": float(score)} for sample_id, score in zip(sample_ids, probability.tolist()))
    frame = pd.DataFrame(rows, columns=["sample_id", "y_score"])
    if frame.empty or frame["sample_id"].duplicated().any():
        raise ValueError("Inference produced empty or duplicate sample predictions")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False, lineterminator="\n")
    return output_path
