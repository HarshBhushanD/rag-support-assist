import json
import time
from collections import Counter
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "tickets.jsonl"
RESULT_PATH = ROOT / "baseline_results.json"
SEED = 42

with DATA_PATH.open(encoding="utf-8") as file:
    rows = [json.loads(line) for line in file if line.strip()]

texts = [row["text"] for row in rows]
labels = [row["label"] for row in rows]

if len(set(labels)) < 2:
    raise ValueError("The dataset must contain at least two classes.")

train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts,
    labels,
    test_size=0.25,
    random_state=SEED,
    stratify=labels,
)

model = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
    LogisticRegression(max_iter=1000, random_state=SEED),
)

start = time.perf_counter()
model.fit(train_texts, train_labels)
training_seconds = time.perf_counter() - start

start = time.perf_counter()
predictions = model.predict(test_texts)
inference_seconds = time.perf_counter() - start

results = {
    "dataset": "synthetic_support_tickets",
    "dataset_size": len(rows),
    "train_size": len(train_texts),
    "test_size": len(test_texts),
    "random_seed": SEED,
    "model": "TF-IDF + Logistic Regression",
    "accuracy": accuracy_score(test_labels, predictions),
    "macro_f1": f1_score(test_labels, predictions, average="macro", zero_division=0),
    "training_seconds": training_seconds,
    "total_test_inference_seconds": inference_seconds,
    "mean_inference_ms_per_ticket": (inference_seconds / len(test_texts) * 1000),
    "class_counts": dict(Counter(labels)),
    "classification_report": classification_report(
        test_labels, predictions, output_dict=True, zero_division=0
    ),
}

RESULT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")

print(f"Dataset: {len(rows)} tickets")
print(f"Train/test: {len(train_texts)}/{len(test_texts)}")
print(f"Accuracy: {results['accuracy']:.3f}")
print(f"Macro F1: {results['macro_f1']:.3f}")
print(f"Training time: {training_seconds:.4f}s")
print(f"Mean inference: {results['mean_inference_ms_per_ticket']:.3f} ms/ticket")
print("\nClassification report:")
print(classification_report(test_labels, predictions, zero_division=0))
print(f"\nResults saved to: {RESULT_PATH}")
