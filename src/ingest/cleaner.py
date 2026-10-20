# import re

# PII_PATTERNS = [
#     # Email addresses
#     (
#         re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
#         "[EMAIL_REDACTED]",
#     ),
#     # Indian mobile numbers, optionally prefixed with +91
#     (
#         re.compile(r"(?<!\d)(?:\+91[\s-]?)?[6-9]\d{9}(?!\d)"),
#         "[PHONE_REDACTED]",
#     ),
#     # US-style phone numbers
#     (
#         re.compile(
#             r"(?<!\d)(?:\+1[\s.-]?)?"
#             r"(?:\(\d{3}\)|\d{3})[\s.-]?\d{3}[\s.-]?\d{4}(?!\d)"
#         ),
#         "[PHONE_REDACTED]",
#     ),
#     # US Social Security number format
#     (
#         re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)"),
#         "[SSN_REDACTED]",
#     ),
# ]


# def clean_text(text: str) -> str:
#     """Normalize whitespace in extracted document text."""
#     if not isinstance(text, str):
#         raise TypeError("text must be a string")

#     return re.sub(r"\s+", " ", text).strip()


# def redact_pii(text: str) -> str:
#     """Redact common PII patterns using regular expressions."""
#     if not isinstance(text, str):
#         raise TypeError("text must be a string")

#     for pattern, replacement in PII_PATTERNS:
#         text = pattern.sub(replacement, text)

#     return text


# def process_text(text: str) -> str:
#     """Clean whitespace, then redact common PII patterns."""
#     return redact_pii(clean_text(text))


import re

from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

PII_PATTERNS = [
    (
        re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
        "[EMAIL_REDACTED]",
    ),
    (
        re.compile(r"(?<!\d)(?:\+91[\s-]?)?[6-9]\d{9}(?!\d)"),
        "[PHONE_REDACTED]",
    ),
    (
        re.compile(
            r"(?<!\d)(?:\+1[\s.-]?)?"
            r"(?:\(\d{3}\)|\d{3})[\s.-]?\d{3}[\s.-]?\d{4}(?!\d)"
        ),
        "[PHONE_REDACTED]",
    ),
    (
        re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)"),
        "[SSN_REDACTED]",
    ),
]


def clean_text(text: str) -> str:
    """Normalize whitespace in text."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    return re.sub(r"\s+", " ", text).strip()


def redact_with_regex(text: str) -> str:
    """Redact known PII patterns using regular expressions."""
    for pattern, replacement in PII_PATTERNS:
        text = pattern.sub(replacement, text)

    return text


# def redact_with_presidio(text: str) -> str:
#     """Detect and redact common PII entities using Presidio."""
#     analyzer = AnalyzerEngine()
#     anonymizer = AnonymizerEngine()

#     results = analyzer.analyze(
#         text=text,
#         language="en",
#     )

#     return anonymizer.anonymize(
#         text=text,
#         analyzer_results=results,
#     ).text


def redact_with_presidio(text: str) -> str:
    """Detect and redact common PII entities using Presidio."""
    results = analyzer.analyze(
        text=text,
        language="en",
        entities=[
            "EMAIL_ADDRESS",
            "PHONE_NUMBER",
            "US_SSN",
            "CREDIT_CARD",
            "IP_ADDRESS",
        ],
    )

    return anonymizer.anonymize(
        text=text,
        analyzer_results=results,
    ).text


def redact_pii(text: str) -> str:
    """Apply regex redaction, then Presidio when available."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    text = redact_with_regex(text)

    try:
        return redact_with_presidio(text)
    except Exception:
        # Keep regex protection if Presidio fails.
        return text


def process_text(text: str) -> str:
    """Normalize text and redact personally identifiable information."""
    return redact_pii(clean_text(text))
