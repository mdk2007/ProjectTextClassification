"""
Training and Evaluation Pipeline for Sentiment Analysis
Trains TF-IDF Vectorizer + Logistic Regression & Multinomial Naive Bayes
Evaluates and saves artifacts for Streamlit UI deployment.
"""

import os
import sys
import json
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

# Ensure project root is in sys.path so 'src' package can be imported from anywhere
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from src.preprocess import preprocess_text
except ImportError:
    from preprocess import preprocess_text

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
RAW_DATA_PATH = os.path.join(DATA_DIR, "IMDB Dataset.csv")
CLEANED_DATA_PATH = os.path.join(DATA_DIR, "cleaned_imdb_dataset.csv")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


def load_and_preprocess_dataset(sample_size: int = None) -> pd.DataFrame:
    """
    Loads dataset, performs NLP preprocessing if not cached, and saves cleaned dataset.
    Supports optional sample_size for rapid training iterations.
    """
    if os.path.exists(CLEANED_DATA_PATH):
        print(f"[*] Found cached preprocessed dataset at: {CLEANED_DATA_PATH}")
        df = pd.read_csv(CLEANED_DATA_PATH)
        if sample_size and len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        return df

    if not os.path.exists(RAW_DATA_PATH):
        print("[*] Local dataset not found. Downloading via kagglehub...")
        import kagglehub
        path = kagglehub.dataset_download("lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")
        raw_src = os.path.join(path, "IMDB Dataset.csv")
        import shutil
        shutil.copy(raw_src, RAW_DATA_PATH)
        print(f"[*] Downloaded and copied to {RAW_DATA_PATH}")

    print(f"[*] Reading raw dataset from {RAW_DATA_PATH}...")
    df = pd.read_csv(RAW_DATA_PATH)
    print(f"[*] Loaded {len(df)} reviews. Class distribution:")
    print(df['sentiment'].value_counts())

    if sample_size and len(df) > sample_size:
        print(f"[*] Sampling {sample_size} records for training...")
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

    print("[*] Running NLP Preprocessing (cleaning, negation-aware stopwords, lemmatization)...")
    t0 = time.time()
    
    # Process in batches for progress reporting
    total = len(df)
    batch_size = 5000
    cleaned_reviews = []
    
    for i in range(0, total, batch_size):
        chunk = df['review'].iloc[i:i + batch_size]
        cleaned_chunk = [preprocess_text(text) for text in chunk]
        cleaned_reviews.extend(cleaned_chunk)
        elapsed = time.time() - t0
        print(f"    - Processed {min(i + batch_size, total)}/{total} reviews ({elapsed:.1f}s)")

    df['cleaned_review'] = cleaned_reviews
    df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})

    # Filter out any empty rows
    df = df[df['cleaned_review'].str.strip() != ''].reset_index(drop=True)

    print(f"[*] Saving cleaned dataset to {CLEANED_DATA_PATH}...")
    df.to_csv(CLEANED_DATA_PATH, index=False)
    print(f"[*] Preprocessing complete in {time.time() - t0:.1f}s.")
    return df


def extract_top_features(vectorizer: TfidfVectorizer, model, feature_names: list, top_n: int = 25) -> dict:
    """Extract top positive and negative indicator words/phrases."""
    if isinstance(model, LogisticRegression):
        coefs = model.coef_[0]
        pos_indices = np.argsort(coefs)[-top_n:][::-1]
        neg_indices = np.argsort(coefs)[:top_n]
        return {
            "model": "Logistic Regression",
            "top_positive": [{"feature": feature_names[i], "weight": round(float(coefs[i]), 4)} for i in pos_indices],
            "top_negative": [{"feature": feature_names[i], "weight": round(float(coefs[i]), 4)} for i in neg_indices],
        }
    elif isinstance(model, MultinomialNB):
        # Calculate log probability ratio: log(P(w|pos) / P(w|neg))
        log_prob_pos = model.feature_log_prob_[1]
        log_prob_neg = model.feature_log_prob_[0]
        ratio = log_prob_pos - log_prob_neg
        pos_indices = np.argsort(ratio)[-top_n:][::-1]
        neg_indices = np.argsort(ratio)[:top_n]
        return {
            "model": "Multinomial Naive Bayes",
            "top_positive": [{"feature": feature_names[i], "weight": round(float(ratio[i]), 4)} for i in pos_indices],
            "top_negative": [{"feature": feature_names[i], "weight": round(float(ratio[i]), 4)} for i in neg_indices],
        }
    return {}


def train_and_evaluate(sample_size: int = None, max_features: int = 10000):
    """
    Main training pipeline:
    1. Preprocesses & splits data
    2. Fits TF-IDF Vectorizer
    3. Trains Logistic Regression & Multinomial Naive Bayes
    4. Evaluates metrics & saves artifacts
    """
    print("=" * 60)
    print("  IMDb Sentiment Analysis - Training & Evaluation Pipeline")
    print("=" * 60)

    # 1. Load Data
    df = load_and_preprocess_dataset(sample_size=sample_size)
    
    # 2. Train/Test Split
    print("\n[*] Splitting into Train (80%) and Test (20%) sets...")
    X = df['cleaned_review'].astype(str)
    y = df['label'].values

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"    Train size: {len(X_train_raw):,} samples")
    print(f"    Test size:  {len(X_test_raw):,} samples")

    # 3. Feature Extraction: TF-IDF
    print(f"\n[*] Vectorizing with TF-IDF (max_features={max_features}, ngram_range=(1,2))...")
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=3
    )
    t0 = time.time()
    X_train_vec = vectorizer.fit_transform(X_train_raw)
    X_test_vec = vectorizer.transform(X_test_raw)
    feature_names = vectorizer.get_feature_names_out()
    print(f"    Vectorization completed in {time.time() - t0:.2f}s")
    print(f"    Vocabulary size: {len(feature_names):,} terms")
    print(f"    Sparse matrix density: {X_train_vec.nnz / (X_train_vec.shape[0] * X_train_vec.shape[1]):.4%}")

    # 4. Train Models
    models = {
        "Logistic Regression": LogisticRegression(
            C=1.0, 
            max_iter=1000, 
            random_state=42,
            solver='lbfgs'
        ),
        "Multinomial Naive Bayes": MultinomialNB(
            alpha=1.0
        )
    }

    metrics_results = {}
    top_features_results = {}

    for name, clf in models.items():
        print(f"\n[*] Training {name}...")
        t0 = time.time()
        clf.fit(X_train_vec, y_train)
        train_time = time.time() - t0

        t0 = time.time()
        y_pred = clf.predict(X_test_vec)
        y_prob = clf.predict_proba(X_test_vec)[:, 1]
        eval_time = time.time() - t0

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred).tolist()

        print(f"    Training time: {train_time:.2f}s | Inference time: {eval_time:.2f}s")
        print(f"    Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
        print(f"    Precision: {prec:.4f}")
        print(f"    Recall:    {rec:.4f}")
        print(f"    F1-Score:  {f1:.4f}")
        print(f"    ROC-AUC:   {roc:.4f}")
        print(f"    Confusion Matrix: {cm}")

        metrics_results[name] = {
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
            "roc_auc": round(float(roc), 4),
            "confusion_matrix": cm,
            "training_time_seconds": round(float(train_time), 3),
            "inference_time_seconds": round(float(eval_time), 3),
            "test_samples": int(len(y_test))
        }

        top_features_results[name] = extract_top_features(vectorizer, clf, feature_names)

    # 5. Save Artifacts
    print("\n[*] Saving artifacts to models directory...")
    joblib.dump(vectorizer, os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib"))
    joblib.dump(models["Logistic Regression"], os.path.join(MODELS_DIR, "logistic_regression.joblib"))
    joblib.dump(models["Multinomial Naive Bayes"], os.path.join(MODELS_DIR, "naive_bayes.joblib"))

    with open(os.path.join(MODELS_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_results, f, indent=2)

    with open(os.path.join(MODELS_DIR, "top_features.json"), "w", encoding="utf-8") as f:
        json.dump(top_features_results, f, indent=2)

    # Save summary metadata
    summary = {
        "dataset_total_samples": len(df),
        "train_samples": len(X_train_raw),
        "test_samples": len(X_test_raw),
        "vocab_size": len(feature_names),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": metrics_results
    }
    with open(os.path.join(MODELS_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[*] All artifacts successfully saved:")
    print(f"    - {os.path.join(MODELS_DIR, 'tfidf_vectorizer.joblib')}")
    print(f"    - {os.path.join(MODELS_DIR, 'logistic_regression.joblib')}")
    print(f"    - {os.path.join(MODELS_DIR, 'naive_bayes.joblib')}")
    print(f"    - {os.path.join(MODELS_DIR, 'metrics.json')}")
    print(f"    - {os.path.join(MODELS_DIR, 'top_features.json')}")
    print("\n[OK] Pipeline completed successfully!")


if __name__ == "__main__":
    # Train on 15,000 reviews for high accuracy (~88%+) and rapid training
    train_and_evaluate(sample_size=15000, max_features=10000)
