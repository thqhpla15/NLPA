"""Pretrained Transformer + linear token-classification head."""
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from datasets import DatasetDict
from transformers import (AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification,
                          Trainer, TrainingArguments)

from alignment import IGNORE_INDEX, LabelAligner
from entity_metrics import EntityMetrics


class TransformerNER:
    def __init__(self, label_names: List[str], model_name: str = "bert-base-cased",
                 strategy: str = "first", max_length: int = 128):
        self.label_names = label_names
        self.id2label = {i: l for i, l in enumerate(label_names)}
        self.label2id = {l: i for i, l in self.id2label.items()}
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True) \
            if "roberta" in model_name else AutoTokenizer.from_pretrained(model_name)
        self.aligner = LabelAligner(label_names, strategy)
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name, num_labels=len(label_names), id2label=self.id2label, label2id=self.label2id)
        self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    # ---------- training ----------
    def tokenize_dataset(self, raw: DatasetDict) -> DatasetDict:
        fn = lambda b: self.aligner.encode_batch(self.tokenizer, b, self.max_length)
        return raw.map(fn, batched=True, remove_columns=raw["train"].column_names)

    def _compute_metrics(self, eval_pred) -> Dict[str, float]:
        logits, labels = eval_pred
        preds = logits.argmax(-1)
        gold_seqs, pred_seqs = [], []
        for p_row, l_row in zip(preds, labels):
            keep = l_row != IGNORE_INDEX
            gold_seqs.append([self.id2label[int(x)] for x in l_row[keep]])
            pred_seqs.append([self.id2label[int(x)] for x in p_row[keep]])
        m = EntityMetrics.overall(gold_seqs, pred_seqs)
        return {"precision": m["precision"], "recall": m["recall"], "f1": m["f1"]}

    def train(self, tokenized: DatasetDict, output_dir: str = "outputs/ner", epochs: int = 3,
              lr: float = 5e-5, batch_size: int = 16, seed: int = 42) -> Trainer:
        args = TrainingArguments(
            output_dir=output_dir, num_train_epochs=epochs, learning_rate=lr,
            per_device_train_batch_size=batch_size, per_device_eval_batch_size=batch_size * 2,
            weight_decay=0.01, eval_strategy="epoch", save_strategy="no", logging_steps=100,
            report_to="none", seed=seed)
        trainer = Trainer(
            model=self.model, args=args, train_dataset=tokenized["train"], eval_dataset=tokenized["validation"],
            data_collator=DataCollatorForTokenClassification(self.tokenizer),
            compute_metrics=self._compute_metrics)
        trainer.train()
        return trainer

    def save(self, path: str):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    # ---------- inference ----------
    @torch.no_grad()
    def predict(self, sents: List[List[str]], batch_size: int = 32) -> List[List[str]]:
        """Word-level tags: the prediction of each word's first subword."""
        self.model.to(self.device).eval()
        results: List[List[str]] = []
        for s in range(0, len(sents), batch_size):
            batch = sents[s:s + batch_size]
            enc = self.tokenizer(batch, is_split_into_words=True, truncation=True,
                                 max_length=self.max_length, padding=True, return_tensors="pt")
            logits = self.model(**{k: v.to(self.device) for k, v in enc.items()}).logits
            pred_ids = logits.argmax(-1).cpu().numpy()
            for i, words in enumerate(batch):
                tags = ["O"] * len(words)  # words cut by truncation default to O
                seen = set()
                for pos, wid in enumerate(enc.word_ids(batch_index=i)):
                    if wid is not None and wid not in seen:
                        seen.add(wid)
                        tags[wid] = self.id2label[int(pred_ids[i, pos])]
                results.append(tags)
        return results

    @torch.no_grad()
    def inspect(self, words: List[str], gold_tags: Optional[List[str]] = None) -> pd.DataFrame:
        """Show tokens, input_ids, attention_mask, word_ids, labels, logits, predictions for one sentence."""
        self.model.to(self.device).eval()
        enc = self.tokenizer(words, is_split_into_words=True, truncation=True,
                             max_length=self.max_length, padding=True, return_tensors="pt")
        logits = self.model(**{k: v.to(self.device) for k, v in enc.items()}).logits[0].cpu()
        word_ids = enc.word_ids(0)
        df = pd.DataFrame({
            "token": self.tokenizer.convert_ids_to_tokens(enc["input_ids"][0]),
            "input_id": enc["input_ids"][0].tolist(),
            "attention_mask": enc["attention_mask"][0].tolist(),
            "word_id": word_ids,
            "prediction": [self.id2label[int(i)] for i in logits.argmax(-1)],
            "max_logit": np.round(logits.max(-1).values.numpy(), 2),
        })
        if gold_tags is not None:
            ids = [self.label2id[t] for t in gold_tags]
            df["label"] = [self.id2label[i] if i != IGNORE_INDEX else "IGNORE"
                           for i in self.aligner.align(word_ids, ids)]
        print(f"logits shape: {tuple(logits.shape)}")
        return df
