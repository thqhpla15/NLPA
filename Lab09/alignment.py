"""Word-level BIO labels -> subword-level labels."""
from typing import Dict, List, Optional

IGNORE_INDEX = -100  # ignored by torch CrossEntropyLoss and by HF token-classification models


class LabelAligner:
    """
    strategy="first"     (Strategy A): first subword gets the word label, the rest get IGNORE_INDEX.
    strategy="propagate" (Strategy B): first subword gets the word label, later subwords get the
                                       I- version of it (B-X -> I-X, I-X -> I-X, O -> O).
    Special tokens ([CLS], [SEP], padding; word_id None) always get IGNORE_INDEX.
    """

    def __init__(self, label_names: List[str], strategy: str = "first"):
        if strategy not in ("first", "propagate"):
            raise ValueError("strategy must be 'first' or 'propagate'")
        self.strategy = strategy
        self.label_names = label_names
        self.label2id = {l: i for i, l in enumerate(label_names)}

    def _continuation_id(self, label_id: int) -> int:
        name = self.label_names[label_id]
        if name.startswith("B-"):
            return self.label2id["I-" + name[2:]]
        return label_id

    def align(self, word_ids: List[Optional[int]], word_label_ids: List[int]) -> List[int]:
        aligned, prev = [], None
        for wid in word_ids:
            if wid is None:
                aligned.append(IGNORE_INDEX)
            elif wid != prev:
                aligned.append(word_label_ids[wid])
            elif self.strategy == "propagate":
                aligned.append(self._continuation_id(word_label_ids[wid]))
            else:
                aligned.append(IGNORE_INDEX)
            prev = wid
        return aligned

    def encode_batch(self, tokenizer, batch: Dict, max_length: int = 128) -> Dict:
        """Use with datasets.map(batched=True). Expects 'tokens' and 'ner_tags' (int ids)."""
        enc = tokenizer(batch["tokens"], is_split_into_words=True, truncation=True, max_length=max_length)
        enc["labels"] = [
            self.align(enc.word_ids(batch_index=i), tags) for i, tags in enumerate(batch["ner_tags"])
        ]
        return enc
