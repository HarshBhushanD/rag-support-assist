import mlflow

from src.evaluation.mlflow_tracker import log_retrieval_experiment


def test_logs_retrieval_metrics_and_artifact(tmp_path, monkeypatch):
    tracking_uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)

    mlflow.set_tracking_uri(tracking_uri)
    results = {
        "dense": {
            "top_k": 2,
            "queries_evaluated": 3,
            "hit_rate_at_k": 0.8,
            "recall_at_k": 0.7,
            "mrr": 0.6,
            "details": [{"id": "q001", "hit": True}],
        },
        "bm25": {
            "top_k": 2,
            "queries_evaluated": 3,
            "hit_rate_at_k": 0.9,
            "recall_at_k": 0.8,
            "mrr": 0.7,
            "details": [{"id": "q001", "hit": True}],
        },
        "hybrid": {
            "top_k": 2,
            "queries_evaluated": 3,
            "hit_rate_at_k": 1.0,
            "recall_at_k": 0.9,
            "mrr": 0.85,
            "details": [{"id": "q001", "hit": True}],
        },
    }

    run_id = log_retrieval_experiment(
        results=results,
        dataset_name="test_dataset.json",
        corpus_name="test_corpus.json",
        experiment_name="test-rag-retrieval",
    )

    client = mlflow.MlflowClient(tracking_uri=tracking_uri)
    run = client.get_run(run_id)

    assert run.data.params["dataset"] == "test_dataset.json"
    assert run.data.params["corpus"] == "test_corpus.json"
    assert run.data.metrics["hybrid_hit_rate_at_k"] == 1.0
    assert run.data.metrics["hybrid_mrr"] == 0.85

    artifacts = client.list_artifacts(run_id)
    assert any(artifact.path == "retrieval_results.json" for artifact in artifacts)
