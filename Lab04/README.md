# Lab 04 — IMDb Sentiment Classification

## 1. Project Overview

This project implements a text classification system to predict whether an IMDb movie review expresses a positive or negative sentiment.

The project explores text preprocessing, TF-IDF features, Word2Vec embeddings, Logistic Regression, model evaluation, n-gram features, error analysis, and sentiment prediction.

## 2. Dataset

- **Dataset:** IMDb Movie Reviews
- **Source:** Hugging Face Datasets (`stanfordnlp/imdb`)
- **Training split:** 25,000 reviews
- **Test split:** 25,000 reviews
- **Unsupervised split:** 50,000 reviews
- **Labels:** Negative (0), Positive (1)

The official training split is divided into training and validation sets using an 80:20 ratio and stratified sampling. The official test split is reserved for final evaluation.

## 3. Project Structure

```text
Lab04/
├── data/
├── output/
├── src/
│   ├── config.py
│   ├── dataset_loader.py
│   ├── dataset_inspector.py
│   ├── data_splitter.py
│   ├── tfidf_classifier.py
│   ├── word2vec_classifier.py
│   ├── model_evaluator.py
│   ├── error_analyzer.py
│   ├── final_evaluator.py
│   ├── ngram_experiment.py
│   ├── model_interpreter.py
│   ├── sentiment_predictor.py
│   ├── word_order_experiment.py
│   └── main.py
├── requirements.txt
└── README.md
```

## 4. Models and Experiments

### Experiment 1: Dataset Inspection

Inspect class distribution, missing values, duplicate reviews, and document lengths.

### Experiment 2: TF-IDF + Logistic Regression

Use TF-IDF features with unigram and bigram features to classify review sentiment.

### Experiment 3: Text Representation Comparison

Compare TF-IDF features with average Word2Vec embeddings using Logistic Regression.

### Experiment 4: Word Order

Compare average Word2Vec representations of sentences with and without the negation word `not`.

### Experiment 5: N-gram Comparison

Compare unigram features against unigram and bigram features.

### Experiment 6: Model Interpretation

Extract the highest positive and negative Logistic Regression feature weights.

### Additional Tasks

Evaluate the model on the official test set, inspect classification errors, and implement sentiment prediction for user-provided text.

## 5. Evaluation Metrics

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

Experimental results and visualizations are saved in the `output/` directory.

## 6. Installation

Install the required packages using:

```bash
pip install -r requirements.txt
```

## 7. Running the Project

Run the following scripts through the configured Python interpreter in IntelliJ IDEA:

- `main.py` — run the main classification experiments.
- `ngram_experiment.py` — compare unigram and bigram features.
- `model_interpreter.py` — extract top positive and negative features.
- `word_order_experiment.py` — run the Word2Vec word-order experiment.
- `final_evaluator.py` — evaluate the model on the official test split.
- `sentiment_predictor.py` — predict sentiment for input text.

Some experiments train their own models and may take additional time.

## 8. Output Files

The `output/` directory contains experimental results, including:

- Class distribution visualization
- Confusion matrices
- Model comparison results
- N-gram comparison results
- Error analysis results
- Final test results
- Top positive and negative features

## 9. AI Assistance

AI tools were used for programming assistance, debugging, and explanations during development. The student is responsible for verifying the implementation, running the experiments, examining the results, and completing independent analysis and reflection.

A detailed AI assistance statement should document the tools used, their purposes, the code generated, the modifications made, and the verification performed.
