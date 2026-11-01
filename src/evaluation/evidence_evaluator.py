import re
from typing import Any


def normalize_evidence(text: str) -> str:
    """Normalize evidence text for matching."""
    return " ".join(re.findall(r"[a-z0-9]+", text.casefold()))


def evidence_is_present(requirement: str, context: str) -> bool:
    """Check whether required evidence is present in the context."""
    normalized_requirement = normalize_evidence(requirement)
    normalized_context = normalize_evidence(context)

    if not normalized_requirement:
        return False

    if normalized_requirement in normalized_context:
        return True

    requirement_tokens = normalized_requirement.split()
    context_tokens = set(normalized_context.split())

    # Allow a meaningful single-token requirement to match
    # when the context contains that exact token.
    if len(requirement_tokens) == 2:
        return requirement_tokens[-1] in context_tokens

    return False


def evaluate_evidence(
    required_evidence: list[str],
    context: str,
) -> dict[str, Any]:
    """Evaluate whether retrieved context contains required evidence.

    This is a lexical matching heuristic, not semantic verification.
    """
    if not isinstance(context, str) or not context.strip():
        return {
            "sufficient": False,
            "reason": "empty_context",
            "evidence_coverage": 0.0,
            "supported_evidence": [],
            "missing_evidence": required_evidence.copy(),
        }

    if not required_evidence:
        return {
            "sufficient": False,
            "reason": "no_required_evidence",
            "evidence_coverage": 0.0,
            "supported_evidence": [],
            "missing_evidence": [],
        }

    supported = []
    missing = []

    for requirement in required_evidence:
        if evidence_is_present(requirement, context):
            supported.append(requirement)
        else:
            missing.append(requirement)

    coverage = len(supported) / len(required_evidence)

    return {
        "sufficient": not missing,
        "reason": (
            "all_required_evidence_found"
            if not missing
            else "missing_required_evidence"
        ),
        "evidence_coverage": coverage,
        "supported_evidence": supported,
        "missing_evidence": missing,
    }
