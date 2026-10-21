from src.evaluation.failure_analysis import analyze_failures


def test_detects_retrieval_miss_and_extra_sources():
    results = {
        "answers": [
            {
                "id": "q001",
                "question": "How do I reset my password?",
                "answerable": True,
                "abstained": False,
                "expected_sources": ["account-help.md"],
                "retrieved_sources": ["database-troubleshooting.md"],
            }
        ]
    }

    report = analyze_failures(results)
    answer = report["answers"][0]

    assert "retrieval_miss" in answer["issues"]
    assert "extra_retrieved_sources" in answer["issues"]


def test_detects_possible_false_abstention():
    results = {
        "answers": [
            {
                "answerable": True,
                "abstained": True,
                "expected_sources": ["account-help.md"],
                "retrieved_sources": ["account-help.md"],
            }
        ]
    }

    report = analyze_failures(results)

    assert "possible_false_abstention" in report["answers"][0]["issues"]


def test_detects_possible_unsupported_answer():
    results = {
        "answers": [
            {
                "answerable": False,
                "abstained": False,
                "expected_sources": [],
                "retrieved_sources": [],
            }
        ]
    }

    report = analyze_failures(results)

    assert "possible_unsupported_answer" in report["answers"][0]["issues"]


def test_correct_abstention_without_expected_sources():
    results = {
        "answers": [
            {
                "answerable": False,
                "abstained": True,
                "expected_sources": [],
                "retrieved_sources": [],
            }
        ]
    }

    report = analyze_failures(results)

    assert report["answers"][0]["issues"] == []


def test_preserves_retrieval_metrics_and_review_flag():
    metrics = {"hybrid": {"hit_rate_at_k": 1.0}}
    results = {
        "retrieval_metrics": metrics,
        "answers": [
            {
                "id": "q001",
                "answerable": True,
                "abstained": False,
                "expected_sources": ["account-help.md"],
                "retrieved_sources": ["account-help.md"],
                "manual_review_required": True,
            }
        ],
    }

    report = analyze_failures(results)

    assert report["retrieval_metrics"] == metrics
    assert report["summary"]["total_answers"] == 1
    assert report["answers"][0]["manual_review_required"] is True
