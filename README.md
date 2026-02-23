# IMDB Movie Review Sentiment Analysis | NLP

Binary sentiment classification on 50,000 IMDB movie reviews using classical ML (TF-IDF + Logistic Regression) and deep learning (Embedding + Dense layers).

## Problem Statement
Classify movie reviews as **Positive** or **Negative** based on the review text. This is a standard NLP benchmark task demonstrating the full text classification pipeline.

## Dataset
| Attribute | Detail |
|---|---|
| File | IMDB Dataset (50k reviews) |
| Records | 50,000 reviews |
| Classes | Positive / Negative (balanced: 25k each) |
| Split | 80% train / 20% test |

## Methodology
1. **EDA & Visualization** — Review length distribution, class balance, word frequency analysis
2. **Text Preprocessing** — HTML tag removal, lowercasing, stopword filtering, lemmatization
3. **Feature Extraction** — TF-IDF vectorization (for ML) / Tokenizer + padding (for DL)
4. **ML Model** — Logistic Regression with TF-IDF features
5. **DL Model** — Embedding(10k, 128) + GlobalAveragePooling + Dense layers
6. **Evaluation** — Accuracy, Precision, Recall, F1-Score, Confusion Matrix

## Results
| Model | Accuracy |
|---|---|
| **Logistic Regression (ML)** | ~85% |
| **Deep Learning (DL)** | ~89% (Epoch 7: 88.95%) |

> DL model outperforms classical ML by ~4 percentage points on this dataset.

## Technologies
`Python` · `scikit-learn` · `TensorFlow/Keras` · `NLTK` · `Pandas` · `NumPy` · `Seaborn` · `Matplotlib` · `joblib`

## File Structure
```
11_IMDB_Sentiment_Analysis_NLP/
├── project_notebook.ipynb   # Main notebook
├── IMDB Dataset.csv         # Dataset
└── models/                  # Saved model and tokenizer
```

## How to Run
```bash
cd 11_IMDB_Sentiment_Analysis_NLP
jupyter notebook project_notebook.ipynb
```
