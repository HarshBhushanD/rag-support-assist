import re
from typing import Any

EVIDENCE_ALIASES = {
    "database port": ("port",),
    "database credentials": ("credentials",),
}

NEGATION_WORDS = {
    "no",
    "not",
    "never",
    "without",
    "unavailable",
    "failed",
}


def normalize_evidence(text: str) -> str:
    """Normalize evidence text for matching."""
    return " ".join(re.findall(r"[a-z0-9]+", text.casefold()))


def _find_phrase_positions(
    phrase: str,
    context_tokens: list[str],
) -> list[int]:
    """Find starting positions of an exact token sequence."""
    phrase_tokens = phrase.split()

    if not phrase_tokens:
        return []

    positions = []

    for index in range(len(context_tokens) - len(phrase_tokens) + 1):
        if context_tokens[index : index + len(phrase_tokens)] == phrase_tokens:
            positions.append(index)

    return positions


def _alias_has_context(
    requirement: str,
    alias: str,
    context: str,
) -> bool:
    """Validate that an alias appears in its required context."""
    if requirement not in EVIDENCE_ALIASES:
        return True

    sentences = re.split(r"(?<=[.!?])\s+|[\n;]+", context)

    if requirement == "database credentials":
        accepted_patterns = (
            r"\bdatabase\s+connection\b",
            r"\bconnection\s+to\s+the\s+database\b",
            r"\bdatabase\s+troubleshooting\b",
        )
    elif requirement == "database port":
        accepted_patterns = (
            r"\bdatabase\s+connection\b",
            r"\bdatabase\s+hostname\b",
            r"\bdatabase\s+troubleshooting\b",
        )
    else:
        return True

    for sentence in sentences:
        normalized_sentence = normalize_evidence(sentence)

        if alias not in normalized_sentence.split():
            continue

        if any(
            re.search(pattern, normalized_sentence) for pattern in accepted_patterns
        ):
            return True

    return False


def _is_negated(
    position: int,
    phrase_length: int,
    context_tokens: list[str],
) -> bool:
    """Detect simple negation near a matched evidence phrase."""
    window_start = max(0, position - 3)
    window_end = min(
        len(context_tokens),
        position + phrase_length + 3,
    )

    nearby_tokens = context_tokens[window_start:window_end]

    return any(token in NEGATION_WORDS for token in nearby_tokens)


def evidence_is_present(requirement: str, context: str) -> bool:
    """Check whether required evidence appears as an exact phrase or alias.

    This is a lexical heuristic, not semantic verification.
    """
    normalized_requirement = normalize_evidence(requirement)
    normalized_context = normalize_evidence(context)

    if not normalized_requirement or not normalized_context:
        return False

    context_tokens = normalized_context.split()

    candidate_phrases = (
        normalized_requirement,
        *EVIDENCE_ALIASES.get(normalized_requirement, ()),
    )

    for phrase in candidate_phrases:
        phrase_tokens = phrase.split()
        positions = _find_phrase_positions(phrase, context_tokens)

        for position in positions:
            # The short "port" alias is valid only when the context
            # also explicitly mentions a database.
            if phrase != normalized_requirement and not _alias_has_context(
                normalized_requirement,
                phrase,
                context,
            ):
                continue

            if not _is_negated(
                position,
                len(phrase_tokens),
                context_tokens,
            ):
                return True

    return False


def evaluate_evidence(
    required_evidence: list[str],
    context: str,
) -> dict[str, Any]:
    """Evaluate whether retrieved context contains required evidence.

    Evidence matching uses exact phrases and explicitly configured aliases.
    It does not perform semantic verification.
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
