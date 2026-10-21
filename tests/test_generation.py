from unittest.mock import patch

import pytest

from src.generation.generator import generate_answer
from src.generation.prompt import build_messages


def test_build_messages_includes_question_and_context():
    chunks = [
        {
            "text": "Check the database hostname and port.",
            "source": "troubleshooting.md",
        }
    ]

    messages = build_messages(
        "How do I troubleshoot a database connection?",
        chunks,
    )

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "database hostname and port" in messages[1]["content"]
    assert "troubleshooting.md" in messages[1]["content"]
    assert "database connection" in messages[1]["content"]


def test_build_messages_handles_empty_context():
    messages = build_messages("How do I log in?", [])

    assert "No relevant context was retrieved." in messages[1]["content"]
    assert "don't have enough information" in messages[0]["content"]


def test_build_messages_rejects_empty_question():
    with pytest.raises(ValueError, match="question must not be empty"):
        build_messages("   ", [])


def test_build_messages_ignores_empty_chunks():
    chunks = [
        {"text": "   ", "source": "empty.md"},
        {"text": "Reset your password.", "source": "auth-help.md"},
    ]

    messages = build_messages("How do I reset my password?", chunks)

    assert "Reset your password." in messages[1]["content"]
    assert "empty.md" not in messages[1]["content"]


def test_generate_answer_returns_answer_and_sources():
    chunks = [
        {
            "text": "Reset your password from account settings.",
            "source": "auth-help.md",
        },
        {
            "text": "Use the password reset link.",
            "source": "login-guide.md",
        },
    ]

    def fake_generator(*args, **kwargs):
        return [{"generated_text": "Reset it from account settings."}]

    with patch(
        "src.generation.generator.load_generator",
        return_value=fake_generator,
    ):
        result = generate_answer(
            "How do I reset my password?",
            chunks,
        )

    assert result["answer"] == "Reset it from account settings."
    assert result["sources"] == ["auth-help.md", "login-guide.md"]


def test_generate_answer_rejects_invalid_token_limit():
    with pytest.raises(ValueError, match="max_new_tokens must be positive"):
        generate_answer("How do I log in?", [], max_new_tokens=0)


def test_generate_answer_supports_chat_message_output():
    def fake_generator(*args, **kwargs):
        return [
            {
                "generated_text": [
                    {
                        "role": "assistant",
                        "content": "Use the login page.",
                    }
                ]
            }
        ]

    with patch(
        "src.generation.generator.load_generator",
        return_value=fake_generator,
    ):
        result = generate_answer("How do I log in?", [])

    assert result["answer"] == "Use the login page."
    assert result["sources"] == []
