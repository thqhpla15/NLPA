"""
Entry point. Run from the lab09/ directory (mark it as "Sources Root" in IntelliJ).

  python run_experiments.py --stages tokenization baseline transformer analysis
  python run_experiments.py --stages transformer analysis --strategy propagate --epochs 3
"""
import argparse

import pandas as pd

from analysis import ErrorAnalyzer, ContextAnalyzer, plot_confusion_matrix, plot_entity_length
from baseline_ner import BaselineNER
from data import ConllData
from entity_metrics import EntityMetrics
from tokenization_analysis import TokenizationAnalyzer
from transformer_ner import TransformerNER


def metrics_rows(model: str, gold, pred):
    ent, tok = EntityMetrics.overall(gold, pred), EntityMetrics.token_level(gold, pred)
    rows = [{"model": model, "level": "entity", **{k: ent[k] for k in ("precision", "recall", "f1")}},
            {"model": model, "level": "token", **{k: tok[k] for k in ("precision", "recall", "f1")},
             "accuracy": tok["accuracy"]}]
    type_rows = [{"model": model, "entity": t, **{k: m[k] for k in ("precision", "recall", "f1")}}
                 for t, m in EntityMetrics.per_type(gold, pred).items()]
    return rows, type_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="+", default=["tokenization", "baseline", "transformer", "analysis"],
                    choices=["tokenization", "baseline", "transformer", "analysis"])
    ap.add_argument("--model", default="bert-base-cased")
    ap.add_argument("--strategy", default="first", choices=["first", "propagate"])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--dataset", default="eriktks/conll2003")
    ap.add_argument("--n-tok-sentences", type=int, default=200)
    args = ap.parse_args()

    data = ConllData(args.dataset)
    train, test = data.sentences("train"), data.sentences("test")
    gold = [s.tags for s in test]
    test_words = [s.words for s in test]

    all_rows, type_rows, preds, length_res = [], [], {}, {}

    if "tokenization" in args.stages:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model)
        ana = TokenizationAnalyzer(tok)
        subset = test[:args.n_tok_sentences]
        ana.sentence_table(subset).to_csv("tokenization_sentences.csv", index=False)
        ana.word_table(subset).to_csv("tokenization_words.csv", index=False)
        for k, v in ana.summary(subset).items():
            print(f"{k}: {v}")

    if "baseline" in args.stages:
        print("Training baseline ...")
        base = BaselineNER().fit(train)
        preds["baseline"] = base.predict(test_words)

    if "transformer" in args.stages:
        ner = TransformerNER(data.label_names, args.model, args.strategy)
        tokenized = ner.tokenize_dataset(data.raw)
        ner.train(tokenized, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)
        ner.save("outputs/ner_model")
        preds["transformer"] = ner.predict(test_words)

    for name, p in preds.items():
        r, t = metrics_rows(name, gold, p)
        all_rows += r
        type_rows += t
        length_res[name] = EntityMetrics.per_length(gold, p)

    if preds:
        pd.DataFrame(all_rows).to_csv("results.csv", index=False)
        pd.DataFrame(type_rows).to_csv("entity_metrics.csv", index=False)
        pd.DataFrame([{"model": m, "length": b, **{k: v[k] for k in ("precision", "recall", "f1")}}
                      for m, res in length_res.items() for b, v in res.items()]).to_csv(
            "entity_length_metrics.csv", index=False)
        print(pd.DataFrame(all_rows).round(4).to_string(index=False))
        print(pd.DataFrame(type_rows).round(4).to_string(index=False))

    if "analysis" in args.stages and preds:
        plot_entity_length(length_res)
        best = preds.get("transformer") or next(iter(preds.values()))
        plot_confusion_matrix(test, best, data.label_names)
        errs = ErrorAnalyzer.collect(test, best)
        errs.to_csv("error_candidates.csv", index=False)
        print(errs["error_type"].value_counts())
        ContextAnalyzer.ambiguous_surface_forms(test).to_csv("ambiguous_surface_forms.csv", index=False)


if __name__ == "__main__":
    main()
