"""NER -> structured information, plus entity-level evaluation of the IE output."""
import json
import re
from typing import Callable, Dict, List, Set, Tuple

from entity_metrics import get_entities, prf

TYPE_NAMES = {"PER": "PERSON", "ORG": "ORGANIZATION", "LOC": "LOCATION", "MISC": "MISC"}
_WORD_RE = re.compile(r"\w+(?:[-']\w+)*|[^\w\s]")

Predictor = Callable[[List[List[str]]], List[List[str]]]  # list of word lists -> list of tag lists


def simple_tokenize(text: str) -> List[str]:
    return _WORD_RE.findall(text)


class InformationExtractor:
    def __init__(self, predictor: Predictor, keep_types=("PER", "ORG", "LOC")):
        self.predictor = predictor
        self.keep_types = set(keep_types)

    def extract(self, text: str) -> Dict[str, List[str]]:
        """Raw text -> {"PERSON": [...], "LOCATION": [...], ...}"""
        words = simple_tokenize(text)
        tags = self.predictor([words])[0]
        out: Dict[str, List[str]] = {TYPE_NAMES[t]: [] for t in TYPE_NAMES if t in self.keep_types}
        for etype, a, b in get_entities(tags):
            if etype in self.keep_types:
                name = " ".join(words[a:b + 1])
                if name not in out[TYPE_NAMES[etype]]:
                    out[TYPE_NAMES[etype]].append(name)
        return out

    def extract_many(self, texts: List[str]) -> List[Dict[str, List[str]]]:
        return [self.extract(t) for t in texts]

    @staticmethod
    def to_json(result: Dict[str, List[str]]) -> str:
        return json.dumps(result, ensure_ascii=False, indent=2)


def evaluate_ie(predicted: List[Dict[str, List[str]]], gold: List[Dict[str, List[str]]]) -> Dict[str, Dict[str, float]]:
    """
    Entity-level P/R/F1 over (sentence_id, type, entity string) triples; exact string match.
    `gold` is the hand-annotated dict per sentence in the same format as extract().
    """
    def triples(dicts) -> Set[Tuple[int, str, str]]:
        return {(i, t, e) for i, d in enumerate(dicts) for t, ents in d.items() for e in ents}

    P, G = triples(predicted), triples(gold)
    result = {"overall": prf(len(P & G), len(P - G), len(G - P))}
    for t in sorted({t for _, t, _ in P | G}):
        Pt, Gt = {x for x in P if x[1] == t}, {x for x in G if x[1] == t}
        result[t] = prf(len(Pt & Gt), len(Pt - Gt), len(Gt - Pt))
    return result
