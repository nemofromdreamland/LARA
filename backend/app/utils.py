import asyncio
import contextvars
import functools

# Holds the request-scoped UUID set by RequestIDMiddleware.
request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default="no-request"
)


def get_request_id() -> str:
    return request_id_var.get()


async def run_sync(fn, *args, **kwargs):
    """Run a blocking callable in the default thread-pool executor."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, functools.partial(fn, *args, **kwargs))
