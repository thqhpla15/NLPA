"""Experiment 1: how many subword tokens does the tokenizer create?"""
from typing import List

import pandas as pd

from data import Sentence


class TokenizationAnalyzer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def _subword_counts(self, words: List[str]) -> List[int]:
        enc = self.tokenizer(words, is_split_into_words=True, add_special_tokens=False)
        counts = [0] * len(words)
        for wid in enc.word_ids():
            if wid is not None:
                counts[wid] += 1
        return counts

    def sentence_table(self, sentences: List[Sentence]) -> pd.DataFrame:
        rows = []
        for i, s in enumerate(sentences):
            c = self._subword_counts(s.words)
            n_words, n_tokens = len(s.words), sum(c)
            rows.append({
                "sent_id": i, "text": s.text, "n_words": n_words, "n_tokens": n_tokens,
                "n_subword_splits": sum(1 for x in c if x > 1), "ratio": n_tokens / n_words,
            })
        return pd.DataFrame(rows)

    def word_table(self, sentences: List[Sentence]) -> pd.DataFrame:
        rows = []
        for i, s in enumerate(sentences):
            c = self._subword_counts(s.words)
            for w, t, n in zip(s.words, s.tags, c):
                rows.append({
                    "sent_id": i, "word": w, "tag": t, "is_entity": t != "O", "n_subwords": n,
                    "pieces": " ".join(self.tokenizer.tokenize(w)),
                    "is_capitalized": w[:1].isupper(), "has_digit": any(ch.isdigit() for ch in w),
                    "length": len(w),
                })
        return pd.DataFrame(rows)

    def summary(self, sentences: List[Sentence]) -> dict:
        st, wt = self.sentence_table(sentences), self.word_table(sentences)
        median = st["n_words"].median()
        short, long_ = st[st["n_words"] <= median], st[st["n_words"] > median]
        ratio = lambda df: df["n_subwords"].mean()
        return {
            "n_sentences": len(st),
            "overall_ratio": st["n_tokens"].sum() / st["n_words"].sum(),
            "short_sentence_ratio": short["n_tokens"].sum() / short["n_words"].sum(),
            "long_sentence_ratio": long_["n_tokens"].sum() / long_["n_words"].sum(),
            "entity_words_avg_subwords": ratio(wt[wt["is_entity"]]),
            "non_entity_words_avg_subwords": ratio(wt[~wt["is_entity"]]),
            "digit_words_avg_subwords": ratio(wt[wt["has_digit"]]) if wt["has_digit"].any() else float("nan"),
            "most_split_words": wt.sort_values("n_subwords", ascending=False).head(15)[["word", "pieces", "tag"]],
        }
