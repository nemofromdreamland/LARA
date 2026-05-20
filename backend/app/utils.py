import asyncio
import functools


async def run_sync(fn, *args, **kwargs):
    """Run a blocking callable in the default thread-pool executor."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, functools.partial(fn, *args, **kwargs))
