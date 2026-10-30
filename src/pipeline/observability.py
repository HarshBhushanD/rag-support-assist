import os

from dotenv import load_dotenv
from langfuse import get_client

load_dotenv()


class NoOpObservation:
    """Stand-in observation that does nothing when tracing is disabled."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def update(self, **kwargs):
        pass


class NoOpLangfuse:
    """Compatible no-op interface for disabled Langfuse tracing."""

    def start_as_current_observation(self, **kwargs):
        return NoOpObservation()


def get_langfuse_client():
    """Return a Langfuse client when credentials are configured."""
    if not os.getenv("LANGFUSE_PUBLIC_KEY") or not os.getenv("LANGFUSE_SECRET_KEY"):
        return NoOpLangfuse()

    return get_client()
