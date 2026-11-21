"""Baseline: hand-crafted token features -> Logistic Regression (one prediction per word, no sequence model)."""
import re
from typing import Dict, List

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

from data import Sentence


def word_shape(w: str) -> str:
    s = re.sub("[A-Z]", "X", w)
    s = re.sub("[a-z]", "x", s)
    s = re.sub("[0-9]", "d", s)
    return re.sub(r"(.)\1{2,}", r"\1\1", s)  # collapse long runs


class BaselineNER:
    def __init__(self, C: float = 1.0, max_iter: int = 200):
        self.vec = DictVectorizer()
        self.clf = LogisticRegression(C=C, max_iter=max_iter)

    @staticmethod
    def _word_feats(words: List[str], i: int) -> Dict:
        w = words[i]
        f = {
            "bias": 1.0, "w.lower": w.lower(), "w.suf3": w[-3:].lower(), "w.suf2": w[-2:].lower(),
            "w.pre3": w[:3].lower(), "w.shape": word_shape(w), "w.istitle": w.istitle(),
            "w.isupper": w.isupper(), "w.isdigit": w.isdigit(), "w.first": i == 0,
        }
        for off in (-1, 1):
            j = i + off
            if 0 <= j < len(words):
                n = words[j]
                f[f"{off}:w.lower"] = n.lower()
                f[f"{off}:w.istitle"] = n.istitle()
                f[f"{off}:w.shape"] = word_shape(n)
            else:
                f[f"{off}:BOUNDARY"] = True
        return f

    def _featurize(self, sents: List[List[str]]) -> List[Dict]:
        return [self._word_feats(ws, i) for ws in sents for i in range(len(ws))]

    def fit(self, train: List[Sentence]) -> "BaselineNER":
        X = self.vec.fit_transform(self._featurize([s.words for s in train]))
        y = [t for s in train for t in s.tags]
        self.clf.fit(X, y)
        return self

    def predict(self, sents: List[List[str]]) -> List[List[str]]:
        flat = self.clf.predict(self.vec.transform(self._featurize(sents)))
        out, k = [], 0
        for ws in sents:
            out.append(list(flat[k:k + len(ws)]))
            k += len(ws)
        return out
