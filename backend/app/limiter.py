from slowapi import Limiter
from starlette.requests import Request

from app.config import settings


def _get_real_ip(request: Request) -> str:
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    return request.client.host


limiter = Limiter(key_func=_get_real_ip, storage_uri=settings.redis_url)
