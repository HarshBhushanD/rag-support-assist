import json
from pathlib import Path

path = Path("data/eval/baseline_results.json")
data = json.loads(path.read_text(encoding="utf-8"))

for mode, metrics in data["retrieval_metrics"].items():
    print(f"\nMode: {mode}")
    print(f"Hit Rate@{metrics['top_k']}: {metrics['hit_rate_at_k']:.3f}")
    print(f"Recall@{metrics['top_k']}: {metrics['recall_at_k']:.3f}")
    print(f"MRR: {metrics['mrr']:.3f}")
