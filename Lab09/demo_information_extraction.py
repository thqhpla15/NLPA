"""Quick IE demo using the fine-tuned model saved by run_experiments.py (outputs/ner_model)."""
from information_extraction import InformationExtractor
from transformer_ner import TransformerNER

LABELS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]

if __name__ == "__main__":
    ner = TransformerNER(LABELS, model_name="outputs/ner_model")
    ie = InformationExtractor(ner.predict)
    print(ie.to_json(ie.extract("Barack Obama visited Paris.")))
