import os

from dotenv import load_dotenv
from langfuse import get_client

load_dotenv()

required_keys = (
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
)

missing_keys = [key for key in required_keys if not os.getenv(key)]
if missing_keys:
    raise RuntimeError(
        f"Missing Langfuse environment variables: {', '.join(missing_keys)}"
    )

langfuse = get_client()

with langfuse.start_as_current_observation(
    as_type="span",
    name="day-20-connection-test",
    input={"task": "Verify Langfuse connection"},
) as span:
    span.update(
        output={
            "status": "success",
            "message": "First trace created successfully",
        }
    )

langfuse.flush()
print("Langfuse trace sent. Check your dashboard.")
