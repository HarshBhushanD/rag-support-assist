import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "tickets.jsonl"
OUTPUT_DIR = ROOT / "lora_model"
RESULT_PATH = ROOT / "lora_results.json"

MODEL_NAME = "distilbert/distilbert-base-uncased"
SEED = 42
EPOCHS = 3
BATCH_SIZE = 4
MAX_LENGTH = 96
LEARNING_RATE = 2e-4

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

with DATA_PATH.open(encoding="utf-8") as file:
    rows = [json.loads(line) for line in file if line.strip()]

labels = sorted({row["label"] for row in rows})
label_to_id = {label: index for index, label in enumerate(labels)}
id_to_label = {index: label for label, index in label_to_id.items()}

texts = [row["text"] for row in rows]
targets = [label_to_id[row["label"]] for row in rows]

train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts,
    targets,
    test_size=0.25,
    random_state=SEED,
    stratify=targets,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(labels),
    id2label=id_to_label,
    label2id=label_to_id,
)

config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=4,
    lora_alpha=8,
    lora_dropout=0.1,
    target_modules=["q_lin", "v_lin"],
    modules_to_save=["pre_classifier", "classifier"],
)
model = get_peft_model(model, config)
model.to("cpu")
model.train()

train_encodings = tokenizer(
    train_texts,
    truncation=True,
    padding=True,
    max_length=MAX_LENGTH,
    return_tensors="pt",
)
train_targets = torch.tensor(train_labels, dtype=torch.long)

optimizer = torch.optim.AdamW(
    (parameter for parameter in model.parameters() if parameter.requires_grad),
    lr=LEARNING_RATE,
)

training_start = time.perf_counter()

for epoch in range(EPOCHS):
    order = torch.randperm(len(train_texts))
    total_loss = 0.0

    for start in range(0, len(order), BATCH_SIZE):
        indices = order[start : start + BATCH_SIZE]
        batch = {key: value[indices] for key, value in train_encodings.items()}
        batch["labels"] = train_targets[indices]

        optimizer.zero_grad()
        output = model(**batch)
        output.loss.backward()
        optimizer.step()
        total_loss += output.loss.item() * len(indices)

    print(f"Epoch {epoch + 1}/{EPOCHS} | loss={total_loss / len(train_texts):.4f}")

training_seconds = time.perf_counter() - training_start

model.eval()
test_encodings = tokenizer(
    test_texts,
    truncation=True,
    padding=True,
    max_length=MAX_LENGTH,
    return_tensors="pt",
)

predictions = []
inference_start = time.perf_counter()

with torch.no_grad():
    for start in range(0, len(test_texts), BATCH_SIZE):
        batch = {
            key: value[start : start + BATCH_SIZE]
            for key, value in test_encodings.items()
        }
        logits = model(**batch).logits
        predictions.extend(logits.argmax(dim=-1).tolist())

inference_seconds = time.perf_counter() - inference_start

true_labels = [id_to_label[value] for value in test_labels]
predicted_labels = [id_to_label[value] for value in predictions]

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

results = {
    "dataset": "synthetic_support_tickets",
    "dataset_size": len(rows),
    "train_size": len(train_texts),
    "test_size": len(test_texts),
    "random_seed": SEED,
    "base_model": MODEL_NAME,
    "method": "LoRA sequence classification",
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "trainable_parameters": sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    ),
    "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
    "accuracy": accuracy_score(true_labels, predicted_labels),
    "macro_f1": f1_score(
        true_labels, predicted_labels, average="macro", zero_division=0
    ),
    "training_seconds": training_seconds,
    "total_test_inference_seconds": inference_seconds,
    "mean_inference_ms_per_ticket": (inference_seconds / len(test_texts) * 1000),
    "classification_report": classification_report(
        true_labels, predicted_labels, output_dict=True, zero_division=0
    ),
}

RESULT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")

print(f"\nAccuracy: {results['accuracy']:.3f}")
print(f"Macro F1: {results['macro_f1']:.3f}")
print(f"Training time: {training_seconds:.2f}s")
print(f"Mean inference: {results['mean_inference_ms_per_ticket']:.2f} ms/ticket")
print("\nClassification report:")
print(classification_report(true_labels, predicted_labels, zero_division=0))
print(f"Adapter saved to: {OUTPUT_DIR}")
print(f"Results saved to: {RESULT_PATH}")
