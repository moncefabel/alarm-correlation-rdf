from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
FIXTURES = Path(__file__).resolve().parent / "fixtures"
NUMBERS = ("9900001", "9900002", "9900003")


@pytest.fixture(scope="session")
def records():
    from parse import parse
    return [parse((FIXTURES / f"{n}.html").read_text(), number=n,
                  url=f"https://userapps.support.sap.com/sap/support/knowledge/en/{n}")
            for n in NUMBERS]


def bow_encoder(dim: int = 512):
    """Deterministic stand-in for a sentence transformer: bag of words."""
    def encode(texts):
        out = np.zeros((len(texts), dim))
        for i, t in enumerate(texts):
            for w in re.findall(r"[a-z0-9]+", t.lower()):
                if w not in {"sap", "the", "of", "a", "is", "and", "alert"}:
                    out[i, hash(w) % dim] += 1.0
        return out
    return encode
