"""Load CoNLL-2003 and expose it as simple Sentence objects."""
from dataclasses import dataclass
from typing import List

from datasets import DatasetDict, load_dataset


@dataclass
class Sentence:
    words: List[str]
    tags: List[str]  # BIO tags as strings, e.g. "B-PER"

    @property
    def text(self) -> str:
        return " ".join(self.words)


class ConllData:
    """Wrapper around the HuggingFace CoNLL-2003 dataset."""

    def __init__(self, dataset_name: str = "eriktks/conll2003"):
        self.raw: DatasetDict = load_dataset(dataset_name)
        self.label_names: List[str] = self.raw["train"].features["ner_tags"].feature.names
        self.label2id = {l: i for i, l in enumerate(self.label_names)}
        self.id2label = {i: l for l, i in self.label2id.items()}

    def sentences(self, split: str) -> List[Sentence]:
        out = []
        for ex in self.raw[split]:
            if not ex["tokens"]:
                continue
            out.append(Sentence(list(ex["tokens"]), [self.id2label[t] for t in ex["ner_tags"]]))
        return out
