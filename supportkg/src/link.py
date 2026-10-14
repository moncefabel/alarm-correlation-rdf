"""Embedding-based entity linking with a Hugging Face sentence encoder.

Rules link what is spelled exactly. Two cases need meaning rather than
spelling:
  - a product string that matches no label of the PPMS stand-in
    ("HANA 2 platform" for "SAP HANA, platform edition 2.0");
  - an article about an alert it never numbers ("Alerts related to HANA memory
    consumption" is about the memory alerts).
Both are nearest-neighbour searches in an embedding space: the mention and
every candidate label are encoded by the same transformer, and the closest
candidates above a threshold are linked. Thresholds are set before looking at
the evaluation set and reported with the results.

Default encoder: sentence-transformers/all-MiniLM-L6-v2 (Hugging Face, runs on
PyTorch, CPU is enough). It is loaded only when first used, so the rest of the
pipeline and the tests run without it.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

Encoder = Callable[[Sequence[str]], np.ndarray]
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def hf_encoder(model_name: str = DEFAULT_MODEL) -> Encoder:
    from sentence_transformers import SentenceTransformer  # optional dependency
    model = SentenceTransformer(model_name)

    def encode(texts: Sequence[str]) -> np.ndarray:
        return model.encode(list(texts), normalize_embeddings=True, convert_to_numpy=True)
    return encode


def _unit(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.where(n == 0, 1, n)


class EmbeddingLinker:
    def __init__(self, product_labels: dict[str, str], alert_names: dict[int, str],
                 encoder: Encoder | None = None, product_threshold: float = 0.80,
                 alert_threshold: float = 0.55, max_alerts: int = 3):
        self._encoder = encoder
        self.product_threshold = product_threshold
        self.alert_threshold = alert_threshold
        self.max_alerts = max_alerts
        self._products = list(product_labels.items())        # (label, iri)
        self._alerts = sorted(alert_names.items())           # (number, name)
        self._pv = self._av = None

    @property
    def encode(self) -> Encoder:
        if self._encoder is None:
            self._encoder = hf_encoder()
        return self._encoder

    def _vecs(self, texts: Sequence[str]) -> np.ndarray:
        return _unit(np.asarray(self.encode(texts), dtype=float))

    def link_product(self, mention: str) -> tuple[str, float] | None:
        if not self._products:
            return None
        if self._pv is None:
            self._pv = self._vecs([lbl for lbl, _ in self._products])
        sims = self._pv @ self._vecs([mention])[0]
        i = int(np.argmax(sims))
        return (self._products[i][1], float(sims[i])) if sims[i] >= self.product_threshold else None

    def link_alerts(self, text: str) -> list[tuple[int, float]]:
        if not self._alerts:
            return []
        if self._av is None:
            self._av = self._vecs([f"SAP HANA alert: {name}" for _, name in self._alerts])
        sims = self._av @ self._vecs([text])[0]
        order = np.argsort(-sims)[: self.max_alerts]
        return [(self._alerts[i][0], float(sims[i])) for i in order if sims[i] >= self.alert_threshold]
