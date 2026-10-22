from typing import Any

import mlflow


def log_retrieval_experiment(
    results: dict[str, Any],
    dataset_name: str,
    corpus_name: str,
    experiment_name: str = "rag-retrieval-evaluation",
) -> str:
    """Log retrieval evaluation metrics and results to MLflow."""
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="retrieval-evaluation") as run:
        mlflow.log_params(
            {
                "dataset": dataset_name,
                "corpus": corpus_name,
                "retrieval_modes": ",".join(sorted(results)),
            }
        )

        for mode, metrics in results.items():
            mlflow.log_params(
                {
                    f"{mode}_top_k": metrics["top_k"],
                    f"{mode}_queries_evaluated": metrics["queries_evaluated"],
                }
            )

            mlflow.log_metrics(
                {
                    f"{mode}_hit_rate_at_k": metrics["hit_rate_at_k"],
                    f"{mode}_recall_at_k": metrics["recall_at_k"],
                    f"{mode}_mrr": metrics["mrr"],
                }
            )

        # Preserve the detailed per-question results for later analysis.
        mlflow.log_dict(results, "retrieval_results.json")

        return run.info.run_id
