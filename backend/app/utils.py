import asyncio
import contextvars
import functools
import re

# Holds the request-scoped UUID set by RequestIDMiddleware.
request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default="no-request"
)


def get_request_id() -> str:
    return request_id_var.get()


def whole_word_match(word: str, text: str) -> bool:
    """True if *word* occurs as a whole word in *text* (case-insensitive)."""
    return bool(re.search(r"\b" + re.escape(word) + r"\b", text, re.IGNORECASE))


async def run_sync(fn, *args, **kwargs):
    """Run a blocking callable in the default thread-pool executor."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, functools.partial(fn, *args, **kwargs))
