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


# ------------------------------------------------------------------------
# Cross-source link: which system views to query when an alert fires.
# Alerts come from knowledge-article titles, views from the help docs; no
# source states the link, so it is predicted and scored against a gold list.

_STOP = {"of", "the", "a", "an", "and", "or", "in", "on", "for", "to", "by", "with",
         "all", "any", "is", "are", "check", "status", "number", "most", "recent",
         "currently", "running", "provides", "displays", "information", "about", "sap",
         "hana", "m", "view", "system", "shows", "lists", "contains"}


def tokens(text: str) -> set[str]:
    import re
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text).replace("_", " ").lower()
    out = set()
    for w in re.findall(r"[a-z][a-z0-9]+", text):
        if w in _STOP:
            continue
        if w.endswith("ies") and len(w) > 5:          # crude plural folding:
            w = w[:-3] + "y"                          # entries -> entry
        elif w.endswith("s") and not w.endswith(("ss", "us", "is")) and len(w) > 4:
            w = w[:-1]                                # services -> service
        out.add(w)
    return out


class ViewLinker:
    """Rank views for an alert name.

    method="lexical": the alert name's content words found in the view,
    each weighted by its rarity across all views (inverse document frequency)
    and counted double when it is in the view's name rather than only in its
    description; normalised to [0, 1] (the baseline).
    method="embedding": cosine similarity of sentence embeddings (Hugging
    Face encoder), on the same two texts.
    Both keep the top `k` views scoring at least `threshold`.
    """

    def __init__(self, views: dict[str, str], method: str = "lexical",
                 encoder: Encoder | None = None, threshold: float | None = None, k: int = 2):
        self.method = method
        self.k = k
        self.threshold = threshold if threshold is not None else (0.35 if method == "lexical" else 0.5)
        self.names = sorted(views)
        self.texts = [f"{n.replace('_', ' ')}: {views[n]}" for n in self.names]
        self._encoder = encoder
        self._vecs = None
        self._name_toks = [tokens(n) for n in self.names]
        self._toks = [tokens(t) for t in self.texts]
        n = len(self.names)
        df: dict[str, int] = {}
        for t in self._toks:
            for w in t:
                df[w] = df.get(w, 0) + 1
        self._idf = {w: float(np.log((n + 1) / (c + 0.5))) for w, c in df.items()}
        self._idf_unseen = float(np.log((n + 1) / 0.5))

    def _encode(self, texts):
        if self._encoder is None:
            self._encoder = hf_encoder()
        return _unit(np.asarray(self._encoder(texts), dtype=float))

    def scores(self, alert_name: str) -> np.ndarray:
        if self.method == "lexical":
            q = tokens(alert_name)
            if not q:
                return np.zeros(len(self.names))
            idf = {w: self._idf.get(w, self._idf_unseen) for w in q}
            full = 2 * sum(idf.values())
            return np.array([sum(idf[w] * (2 if w in nt else 1) for w in q & t) / full
                             for nt, t in zip(self._name_toks, self._toks)])
        if self._vecs is None:
            self._vecs = self._encode(self.texts)
        return self._vecs @ self._encode([f"SAP HANA alert: {alert_name}"])[0]

    def link(self, alert_name: str) -> list[tuple[str, float]]:
        s = self.scores(alert_name)
        order = np.argsort(-s, kind="stable")[: self.k]
        return [(self.names[i], float(s[i])) for i in order if s[i] >= self.threshold]
