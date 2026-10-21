from collections import Counter
from typing import Any


def analyze_failures(results: dict[str, Any]) -> dict[str, Any]:
    """Identify potential RAG evaluation failures for manual review."""
    answers = results.get("answers", [])
    analyzed_answers = []
    issue_counts: Counter[str] = Counter()

    for item in answers:
        issues: list[str] = []

        answerable = item.get("answerable")
        abstained = item.get("abstained")
        expected_sources = set(item.get("expected_sources", []))
        retrieved_sources = set(item.get("retrieved_sources", []))

        if answerable is True and abstained is True:
            issues.append("possible_false_abstention")

        if answerable is False and abstained is False:
            issues.append("possible_unsupported_answer")

        if expected_sources:
            if not expected_sources.intersection(retrieved_sources):
                issues.append("retrieval_miss")

            if retrieved_sources - expected_sources:
                issues.append("extra_retrieved_sources")

        issue_counts.update(issues)

        analyzed_answers.append(
            {
                **item,
                "issues": issues,
                "manual_review_required": bool(
                    issues or item.get("manual_review_required", False)
                ),
            }
        )

    return {
        "summary": {
            "total_answers": len(answers),
            "answers_requiring_review": sum(
                item["manual_review_required"] for item in analyzed_answers
            ),
            "issue_counts": dict(sorted(issue_counts.items())),
        },
        "answers": analyzed_answers,
        "retrieval_metrics": results.get("retrieval_metrics", {}),
        "limitations": [
            "Issue flags are heuristics and require manual review.",
            "Extra retrieved sources are not necessarily irrelevant.",
            "Abstention flags depend on the correctness of the input labels.",
        ],
    }
