"""Token-level and entity-level metrics for BIO-tagged sequences (no external deps)."""
from collections import defaultdict
from typing import Dict, List, Sequence, Set, Tuple

Entity = Tuple[str, int, int]  # (type, start, end) with end inclusive


def get_entities(tags: Sequence[str]) -> List[Entity]:
    """Extract entity spans from a BIO sequence (an I- without a matching open entity starts a new one)."""
    entities: List[Entity] = []
    etype, start = None, -1
    for i, tag in enumerate(list(tags) + ["O"]):
        starts_new = tag.startswith("B-") or (tag.startswith("I-") and tag[2:] != etype)
        if tag == "O" or starts_new:
            if etype is not None:
                entities.append((etype, start, i - 1))
            etype = None
            if tag != "O":
                etype, start = tag[2:], i
    return entities


def prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def length_bucket(length: int) -> str:
    return str(length) if length < 4 else "4+"


class EntityMetrics:
    """Exact-match entity-level evaluation (type AND boundary must match)."""

    @staticmethod
    def _counts(gold_seqs, pred_seqs, keep=lambda e: True) -> Tuple[int, int, int]:
        tp = fp = fn = 0
        for g, p in zip(gold_seqs, pred_seqs):
            gs: Set[Entity] = {e for e in get_entities(g) if keep(e)}
            ps: Set[Entity] = {e for e in get_entities(p) if keep(e)}
            tp += len(gs & ps)
            fp += len(ps - gs)
            fn += len(gs - ps)
        return tp, fp, fn

    @classmethod
    def overall(cls, gold_seqs, pred_seqs) -> Dict[str, float]:
        return prf(*cls._counts(gold_seqs, pred_seqs))

    @classmethod
    def per_type(cls, gold_seqs, pred_seqs, types=("PER", "ORG", "LOC", "MISC")) -> Dict[str, Dict[str, float]]:
        return {t: prf(*cls._counts(gold_seqs, pred_seqs, keep=lambda e, t=t: e[0] == t)) for t in types}

    @classmethod
    def per_length(cls, gold_seqs, pred_seqs, buckets=("1", "2", "3", "4+")) -> Dict[str, Dict[str, float]]:
        def keep_b(b):
            return lambda e: length_bucket(e[2] - e[1] + 1) == b
        return {b: prf(*cls._counts(gold_seqs, pred_seqs, keep=keep_b(b))) for b in buckets}

    @staticmethod
    def token_level(gold_seqs, pred_seqs) -> Dict[str, float]:
        """Micro P/R/F1 over non-O tokens (exact tag match) + accuracy over all tokens."""
        tp = fp = fn = correct = total = 0
        for g, p in zip(gold_seqs, pred_seqs):
            for a, b in zip(g, p):
                total += 1
                correct += a == b
                if a != "O" and a == b:
                    tp += 1
                elif b != "O" and a != b:
                    fp += 1
                if a != "O" and a != b:
                    fn += 1
        out = prf(tp, fp, fn)
        out["accuracy"] = correct / total if total else 0.0
        return out
