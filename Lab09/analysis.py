"""Experiments 3-6: error analysis, entity length plot, confusion matrix, context analysis."""
from collections import defaultdict
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix

from data import Sentence
from entity_metrics import get_entities


class ErrorAnalyzer:
    """Collect candidate errors from gold vs predicted BIO tags. YOU still choose the examples and explain them."""

    @staticmethod
    def collect(sentences: List[Sentence], pred_seqs: List[List[str]]) -> pd.DataFrame:
        rows = []
        span_text = lambda ws, e: " ".join(ws[e[1]:e[2] + 1])
        overlaps = lambda a, b: not (a[2] < b[1] or b[2] < a[1])
        for i, (s, pred) in enumerate(zip(sentences, pred_seqs)):
            g, p = set(get_entities(s.tags)), set(get_entities(pred))
            for e in g - p:
                ov = [q for q in p if overlaps(e, q)]
                if not ov:
                    etype = "missing_entity"
                elif any(q[1:] == e[1:] for q in ov):
                    etype = "wrong_entity_type"
                else:
                    etype = "wrong_boundary"
                rows.append({"sent_id": i, "text": s.text, "gold": f"{span_text(s.words, e)} [{e[0]}]",
                             "prediction": " | ".join(f"{span_text(s.words, q)} [{q[0]}]" for q in ov) or "-",
                             "error_type": etype, "kind": "FN"})
            for e in p - g:
                if not any(overlaps(e, q) for q in g):
                    rows.append({"sent_id": i, "text": s.text, "gold": "-",
                                 "prediction": f"{span_text(s.words, e)} [{e[0]}]",
                                 "error_type": "spurious_entity", "kind": "FP"})
        return pd.DataFrame(rows)

    @staticmethod
    def train_frequency(train: List[Sentence]) -> Dict[str, int]:
        """How often each entity surface form occurs in train (to check rare-entity errors)."""
        freq: Dict[str, int] = defaultdict(int)
        for s in train:
            for e in get_entities(s.tags):
                freq[" ".join(s.words[e[1]:e[2] + 1])] += 1
        return freq


class ContextAnalyzer:
    @staticmethod
    def ambiguous_surface_forms(sentences: List[Sentence], min_types: int = 2) -> pd.DataFrame:
        """Gold entity strings that appear with >= min_types different entity types."""
        seen: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))
        for i, s in enumerate(sentences):
            for t, a, b in get_entities(s.tags):
                seen[" ".join(s.words[a:b + 1])][t].append(i)
        rows = [{"surface": k, "types": sorted(v), "examples": {t: ids[:2] for t, ids in v.items()}}
                for k, v in seen.items() if len(v) >= min_types]
        return pd.DataFrame(rows)


def plot_entity_length(per_length: Dict[str, Dict[str, Dict[str, float]]], path: str = "entity_length_analysis.png"):
    """per_length: {model_name: EntityMetrics.per_length(...)}"""
    fig, ax = plt.subplots(figsize=(6, 4))
    buckets = ["1", "2", "3", "4+"]
    width = 0.8 / len(per_length)
    for k, (name, res) in enumerate(per_length.items()):
        ax.bar([i + k * width for i in range(4)], [res[b]["f1"] for b in buckets], width, label=name)
    ax.set_xticks([i + width * (len(per_length) - 1) / 2 for i in range(4)])
    ax.set_xticklabels(buckets)
    ax.set_xlabel("Entity length (tokens)")
    ax.set_ylabel("Entity-level F1")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(sentences: List[Sentence], pred_seqs: List[List[str]], labels: List[str],
                          path: str = "confusion_matrix.png", normalize: bool = True):
    gold = [t for s in sentences for t in s.tags]
    pred = [t for p in pred_seqs for t in p]
    cm = confusion_matrix(gold, pred, labels=labels, normalize="true" if normalize else None)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Gold")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{cm[i, j]:.2f}" if normalize else int(cm[i, j]), ha="center", va="center",
                    fontsize=7, color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
