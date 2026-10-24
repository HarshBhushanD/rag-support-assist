import re

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "was",
    "were",
    "with",
}


def normalize_claim(text: str) -> str:
    """Normalize text for deterministic comparison."""
    return " ".join(re.findall(r"[a-z0-9]+", text.casefold()))


def evaluate_groundedness(
    answer: str,
    context: str,
    min_token_coverage: float = 0.8,
) -> dict[str, bool | str | float]:
    """Estimate lexical support using content-token coverage.

    This is a heuristic, not a semantic entailment or fact-verification model.
    """
    normalized_answer = normalize_claim(answer)
    normalized_context = normalize_claim(context)

    if not normalized_answer:
        return {
            "grounded": False,
            "reason": "empty_answer",
            "token_coverage": 0.0,
        }

    if not normalized_context:
        return {
            "grounded": False,
            "reason": "empty_context",
            "token_coverage": 0.0,
        }

    answer_tokens = {
        token for token in normalized_answer.split() if token not in STOPWORDS
    }
    context_tokens = set(normalized_context.split())

    if not answer_tokens:
        return {
            "grounded": False,
            "reason": "no_content_tokens",
            "token_coverage": 0.0,
        }

    coverage = len(answer_tokens & context_tokens) / len(answer_tokens)
    grounded = coverage >= min_token_coverage

    return {
        "grounded": grounded,
        "reason": (
            "sufficient_content_token_coverage"
            if grounded
            else "insufficient_content_token_coverage"
        ),
        "token_coverage": coverage,
    }
