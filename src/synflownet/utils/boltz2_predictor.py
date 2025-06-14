from __future__ import annotations

import pathlib
from typing import List

import torch


class Boltz2Predictor:
    """Wrapper for Boltz-2 binding affinity predictor."""

    def __init__(self, model_path: str, device: str = "cpu") -> None:
        self.device = device
        self.model_path = pathlib.Path(model_path)
        self._model = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            import boltz
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "Boltz-2 library is required for affinity prediction"
            ) from exc
        self._model = boltz.Boltz2.load_from_checkpoint(str(self.model_path))
        self._model.to(torch.device(self.device))
        self._model.eval()

    def predict(self, protein_sequence: str, smiles: List[str]) -> List[float]:
        if self._model is None:
            raise RuntimeError("Boltz-2 model not loaded")
        with torch.no_grad():
            preds = self._model.predict(protein_sequence, smiles)
        return [float(p) for p in preds]
