from functools import lru_cache

from transformers import pipeline

from src.generation.prompt import build_messages

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


@lru_cache(maxsize=1)
def load_generator():
    """Load and cache the local text-generation pipeline."""
    return pipeline(
        task="text-generation",
        model=MODEL_NAME,
        device="cpu",
    )


def generate_answer(
    question: str,
    context_chunks: list[dict],
    max_new_tokens: int = 160,
) -> dict:
    """Generate a grounded answer and return its source filenames."""
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")

    messages = build_messages(question, context_chunks)
    generator = load_generator()

    result = generator(
        messages,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        return_full_text=False,
    )

    generated_text = result[0]["generated_text"]

    if isinstance(generated_text, list):
        answer = generated_text[-1]["content"].strip()
    else:
        answer = generated_text.strip()
    sources = list(
        dict.fromkeys(
            chunk.get("source", "unknown")
            for chunk in context_chunks
            if chunk.get("text", "").strip()
        )
    )

    # Remove fallback commentary appended after an otherwise usable answer.
    fallback_marker = "\nFallback:"
    marker_index = answer.casefold().find(fallback_marker.casefold())

    if marker_index >= 0:
        answer = answer[:marker_index].strip()

    return {
        "answer": answer,
        "sources": sources,
    }
