import os

from dotenv import load_dotenv
from langfuse import get_client

load_dotenv()


def get_langfuse_client():
    """Return the configured Langfuse client."""
    if not os.getenv("LANGFUSE_PUBLIC_KEY"):
        raise RuntimeError("LANGFUSE_PUBLIC_KEY is not configured")

    if not os.getenv("LANGFUSE_SECRET_KEY"):
        raise RuntimeError("LANGFUSE_SECRET_KEY is not configured")

    return get_client()
