"""Integration tests for /chat/stream beyond SSE framing.

Covers the behaviours test_chat_route.py's framing tests leave open:
server-side history persistence after [DONE], Redis-down → 503 before any
bytes are streamed, and mid-stream pipeline failure semantics.
"""

import json
from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, patch

import pytest
import redis.exceptions
from fastapi.testclient import TestClient

import app.services.session_store as _ss
from app.services.session_store import get_history


@pytest.fixture
def session_id(client: TestClient) -> str:
    resp = client.post("/session")
    assert resp.status_code == 200
    client.headers["X-Session-Token"] = resp.json()["session_token"]
    return resp.json()["session_id"]


def _make_stream(*payloads: str):
    async def _gen() -> AsyncGenerator[str, None]:
        for p in payloads:
            yield p

    return _gen()


# ---------------------------------------------------------------------------
# History persistence
# ---------------------------------------------------------------------------


@patch("app.routes.chat.answer_stream")
async def test_chat_stream_appends_history_after_done(
    mock_stream, client: TestClient, session_id: str
):
    sources_payload = "[SOURCES]" + json.dumps({"sources": []})
    mock_stream.return_value = _make_stream(
        "Hello ", "world", sources_payload, "[DONE]"
    )

    with client.stream(
        "POST",
        "/chat/stream",
        json={"session_id": session_id, "question": "What are the side effects?"},
    ) as resp:
        assert resp.status_code == 200
        b"".join(resp.iter_bytes())  # consume the full stream

    turns = await get_history(session_id)
    assert [(t.role, t.content) for t in turns] == [
        ("user", "What are the side effects?"),
        ("assistant", "Hello world"),  # concatenated tokens, sources/done excluded
    ]


@patch("app.routes.chat.answer_stream")
async def test_chat_stream_passes_prior_history_to_pipeline(
    mock_stream, client: TestClient, session_id: str
):
    """The history handed to answer_stream is the server-side Redis history."""
    sources_payload = "[SOURCES]" + json.dumps({"sources": []})
    mock_stream.side_effect = lambda *a, **k: _make_stream(
        "answer", sources_payload, "[DONE]"
    )

    for question in ("First question?", "Second question?"):
        with client.stream(
            "POST",
            "/chat/stream",
            json={"session_id": session_id, "question": question},
        ) as resp:
            b"".join(resp.iter_bytes())

    first_call_history = mock_stream.call_args_list[0].args[2]
    second_call_history = mock_stream.call_args_list[1].args[2]
    assert first_call_history == []
    assert [(h["role"], h["content"]) for h in second_call_history] == [
        ("user", "First question?"),
        ("assistant", "answer"),
    ]


# ---------------------------------------------------------------------------
# Redis-down → 503 before streaming starts
# ---------------------------------------------------------------------------


@patch("app.routes.chat.answer_stream")
async def test_chat_stream_redis_down_returns_503(
    mock_stream, client: TestClient, session_id: str
):
    """History is read before the StreamingResponse is built, so a Redis
    outage after session verification yields a proper 503, not a broken
    200 stream."""
    mock_stream.return_value = _make_stream("never streamed")
    # Session owner lookup (hget) still works; only the history read fails.
    _ss._redis.lrange = AsyncMock(
        side_effect=redis.exceptions.ConnectionError("connection refused")
    )

    response = client.post(
        "/chat/stream",
        json={"session_id": session_id, "question": "What are the side effects?"},
    )
    assert response.status_code == 503
    assert "Storage unavailable" in response.json()["detail"]
    mock_stream.assert_not_called()


# ---------------------------------------------------------------------------
# Mid-stream pipeline failure
# ---------------------------------------------------------------------------


@patch("app.routes.chat.answer_stream")
async def test_chat_stream_midstream_failure_truncates_without_done(
    mock_stream, client: TestClient, session_id: str
):
    """If the pipeline raises after tokens were sent, the stream ends without
    an `event: done` frame and no history is persisted — the frontend treats
    a missing done event as an aborted answer."""

    async def _failing_gen() -> AsyncGenerator[str, None]:
        yield "partial "
        raise RuntimeError("provider exploded mid-stream")

    mock_stream.return_value = _failing_gen()

    received = b""
    with pytest.raises(RuntimeError, match="provider exploded"):
        with client.stream(
            "POST",
            "/chat/stream",
            json={"session_id": session_id, "question": "What are the side effects?"},
        ) as resp:
            assert resp.status_code == 200
            for chunk in resp.iter_bytes():
                received += chunk

    text = received.decode()
    assert "event: done" not in text
    assert await get_history(session_id) == []


# ---------------------------------------------------------------------------
# Mid-stream provider failover (reset event)
# ---------------------------------------------------------------------------


@patch("app.routes.chat.answer_stream")
async def test_chat_stream_reset_discards_partial_tokens(
    mock_stream, client: TestClient, session_id: str
):
    """A STREAM_RESET sentinel from the pipeline becomes an `event: reset`
    frame, and only the regenerated answer is persisted to history — the
    partial pre-failover tokens are discarded."""
    from app.services.llm_client import STREAM_RESET

    sources_payload = "[SOURCES]" + json.dumps({"sources": []})
    mock_stream.return_value = _make_stream(
        "Partial ", "garbage", STREAM_RESET, "Clean answer.", sources_payload, "[DONE]"
    )

    received = b""
    with client.stream(
        "POST",
        "/chat/stream",
        json={"session_id": session_id, "question": "What are the side effects?"},
    ) as resp:
        assert resp.status_code == 200
        for chunk in resp.iter_bytes():
            received += chunk

    text = received.decode()
    assert "event: reset" in text
    # The sentinel itself is never forwarded as token data.
    assert json.dumps(STREAM_RESET) not in text

    turns = await get_history(session_id)
    assert [(t.role, t.content) for t in turns] == [
        ("user", "What are the side effects?"),
        ("assistant", "Clean answer."),  # pre-reset tokens excluded
    ]
