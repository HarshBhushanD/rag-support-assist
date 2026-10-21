from src.generation.generator import generate_answer

if __name__ == "__main__":
    chunks = [
        {
            "text": "Support hours are Monday to Friday, 9 AM to 5 PM.",
            "source": "support-hours.md",
        }
    ]

    result = generate_answer(
        question="What is the database administrator's phone number?",
        context_chunks=chunks,
        max_new_tokens=100,
    )

    print("\n--- Generated Answer ---")
    print(result["answer"])
    print("\n--- Sources ---")
    print(result["sources"])
