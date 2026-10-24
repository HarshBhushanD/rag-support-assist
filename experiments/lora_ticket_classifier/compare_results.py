import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

with (ROOT / "baseline_results.json").open(encoding="utf-8") as f:
    baseline = json.load(f)

with (ROOT / "lora_results.json").open(encoding="utf-8") as f:
    lora = json.load(f)

report = {
    "dataset": "synthetic_support_tickets",
    "test_size": baseline["test_size"],
    "baseline": {
        "model": baseline["model"],
        "accuracy": baseline["accuracy"],
        "macro_f1": baseline["macro_f1"],
        "training_seconds": baseline["training_seconds"],
        "mean_inference_ms_per_ticket": baseline["mean_inference_ms_per_ticket"],
    },
    "lora": {
        "model": lora["method"],
        "accuracy": lora["accuracy"],
        "macro_f1": lora["macro_f1"],
        "training_seconds": lora["training_seconds"],
        "mean_inference_ms_per_ticket": lora["mean_inference_ms_per_ticket"],
        "trainable_parameters": lora["trainable_parameters"],
        "total_parameters": lora["total_parameters"],
    },
    "difference_lora_minus_baseline": {
        "accuracy": lora["accuracy"] - baseline["accuracy"],
        "macro_f1": lora["macro_f1"] - baseline["macro_f1"],
        "training_seconds": (lora["training_seconds"] - baseline["training_seconds"]),
        "mean_inference_ms_per_ticket": (
            lora["mean_inference_ms_per_ticket"]
            - baseline["mean_inference_ms_per_ticket"]
        ),
    },
    "limitations": [
        "Only 40 synthetic examples were used.",
        "The held-out test set contains only 10 examples.",
        "Results are exploratory and do not establish real-world performance.",
        "Timing measurements are from a CPU-only local environment.",
    ],
}

output = ROOT / "comparison_results.json"
output.write_text(json.dumps(report, indent=2), encoding="utf-8")

print(json.dumps(report, indent=2))
print(f"\nSaved comparison to: {output}")
