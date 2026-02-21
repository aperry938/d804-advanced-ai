# Transformer Sentiment Analysis & Naive Bayes Spam Classifier

Two NLP classification systems built from scratch: a custom Transformer encoder for IMDB sentiment analysis and a Multinomial Naive Bayes classifier for SMS spam detection, each with systematic hyperparameter optimization.

## Overview

This project implements two distinct approaches to text classification, demonstrating both deep learning and probabilistic machine learning methods.

**Model 1 — Transformer Sentiment Analyzer:** A custom Transformer encoder architecture classifies IMDB movie reviews as positive or negative. The model implements multi-head self-attention with learned positional embeddings, enabling it to capture long-range dependencies in review text. Systematic optimization tunes learning rate, architecture depth, and regularization.

**Model 2 — Naive Bayes Spam Classifier:** A Multinomial Naive Bayes classifier with TF-IDF vectorization detects spam in SMS messages. The probabilistic approach leverages Bayes' theorem with the naive independence assumption, optimized through smoothing parameter tuning, feature selection, and vectorization method comparison.

## Key Techniques

- Custom Transformer encoder with multi-head self-attention and positional encoding
- TF-IDF and Count vectorization with n-gram feature extraction
- Multinomial Naive Bayes with Laplace smoothing
- Systematic hyperparameter optimization (learning rate, architecture, regularization)
- Comprehensive evaluation: accuracy, precision, recall, F1, ROC-AUC, confusion matrices

## Project Structure

```
d804-advanced-ai/
├── transformer_sentiment.py        — Transformer model: build, train, evaluate on IMDB
├── naive_bayes_spam.py             — Naive Bayes model: TF-IDF + MNB on SMS spam
├── transformer_optimization.py     — Transformer hyperparameter tuning pipeline
├── spam_optimization.py            — NB optimization: alpha, features, vectorizer
├── datasets_info.txt               — Dataset sources and licensing
├── sentiment_training_history.png              — Transformer training curves (accuracy/loss)
├── spam_classifier_results.png                 — NB ROC curve and confusion matrix
├── spam_optimization_comparison.png            — Baseline vs optimized performance comparison
├── spam_optimization_results.json              — Optimization metrics (JSON)
├── spam_classifier_model.joblib                — Trained NB model artifact
└── spam_classifier_vectorizer.joblib           — Fitted TF-IDF vectorizer artifact
```

## Results

### Transformer Sentiment Analyzer (IMDB)
- **Architecture:** 2 Transformer blocks, 4 attention heads, 128-dim embeddings
- **Vocabulary:** 10,000 words, sequence length 200
- **Optimization:** Grid search over learning rate (1e-5 to 1e-3), architecture variants, and dropout rates

### Naive Bayes Spam Classifier (SMS)
| Metric    | Baseline | Optimized |
|-----------|----------|-----------|
| Accuracy  | 96.6%    | 97.1%     |
| Precision | 100%     | 94.0%     |
| Recall    | 74.5%    | 83.9%     |
| F1-Score  | 85.4%    | 88.7%     |

The optimized model reduced features from 5,000 to 1,000 while improving F1 by 3.8%, achieving better recall without significant precision loss.

## How to Run

### Prerequisites
```bash
pip install tensorflow numpy pandas scikit-learn matplotlib joblib
```

### Sentiment Analyzer
```bash
# Train and evaluate the Transformer model
python transformer_sentiment.py

# Run hyperparameter optimization
python transformer_optimization.py
```

### Spam Classifier
```bash
# Train and evaluate the Naive Bayes model
python naive_bayes_spam.py

# Run optimization experiments
python spam_optimization.py
```

## Datasets

- **IMDB Movie Reviews** — 50,000 reviews (Maas et al., 2011, Stanford AI Lab). Loaded via `tensorflow.keras.datasets.imdb`.
- **SMS Spam Collection** — 5,574 SMS messages (UCI Machine Learning Repository, CC BY 4.0).

## Author

Anthony Perry — M.S. Computer Science, Western Governors University
