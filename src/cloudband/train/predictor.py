"""Wraps a trained torch model as the Predictor contract pipelines/cloudsen12.py
expects."""

from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray

from cloudband.pipelines.phase0 import select_rgn
from cloudband.train.normalize import dynamic_z_score

CLASS_AXIS = 1


def build_predictor(model: torch.nn.Module, device: str | None = None):
    """Wrap a trained model so it accepts a full band stack and predicts hard classes.

    Applies the same preprocessing training uses: select the R-G-NIR bands,
    then dynamic Z-score normalization, before the model ever sees the data.
    device defaults to cuda when available, so a model trained on GPU does
    not silently fall back to a much slower CPU evaluation.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    def predictor(stack: NDArray[np.integer]) -> NDArray[np.integer]:
        image = dynamic_z_score(select_rgn(stack).astype(np.float32))
        tensor = torch.from_numpy(image).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(tensor)
        prediction = logits.argmax(dim=CLASS_AXIS).squeeze(0).cpu().numpy()
        return prediction.astype(np.int64)

    return predictor