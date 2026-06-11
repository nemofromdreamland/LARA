from slowapi import Limiter
from starlette.requests import Request

from app.config import settings


def _get_real_ip(request: Request) -> str:
    # X-Real-IP is set by our nginx from $remote_addr and cannot be forwarded
    # by the client through the proxy. X-Forwarded-For is NOT trusted: nginx
    # appends to it, so the first entry stays client-controlled (spoofable).
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip.strip()
    return request.client.host


limiter = Limiter(key_func=_get_real_ip, storage_uri=settings.redis_url)
