"""CI-safe unit tests for app.services.embedder.

Mirror of test_embedder.py with the model layer patched out, so the
async/executor plumbing of embed() stays covered when the real-model
tests are excluded via `pytest -m "not ml"`.
"""

from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import app.services.embedder as embedder_module
from app.services.embedder import embed, is_model_loaded, preload_model

_EMBEDDING_DIM = 768


def _fake_vectors(texts: list[str]) -> list[list[float]]:
    """Deterministic per-text vectors: text i maps to [i, i, ..., i]."""
    return [[float(i)] * _EMBEDDING_DIM for i, _ in enumerate(texts)]


# ---------------------------------------------------------------------------
# embed — plumbing
# ---------------------------------------------------------------------------


async def test_embed_returns_encode_sync_output():
    with patch("app.services.embedder._encode_sync", side_effect=_fake_vectors):
        result = await embed(["hello world"])
    assert isinstance(result, list)
    assert result == [[0.0] * _EMBEDDING_DIM]


async def test_embed_passes_texts_through_in_order():
    texts = ["first", "second", "third"]
    with patch("app.services.embedder._encode_sync", side_effect=_fake_vectors):
        result = await embed(texts)
    assert len(result) == 3
    assert [v[0] for v in result] == [0.0, 1.0, 2.0]


async def test_embed_uses_supplied_executor():
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="test-embed")
    used_threads: list[str] = []

    def _fake_encode(texts):
        import threading

        used_threads.append(threading.current_thread().name)
        return _fake_vectors(texts)

    try:
        with patch("app.services.embedder._encode_sync", side_effect=_fake_encode):
            await embed(["hello world"], executor)
    finally:
        executor.shutdown(wait=False)

    assert used_threads and used_threads[0].startswith("test-embed")


async def test_embed_labels_metric_with_source():
    with patch("app.services.embedder._encode_sync", side_effect=_fake_vectors):
        result = await embed(["q"], source="chat")
    assert result == [[0.0] * _EMBEDDING_DIM]


# ---------------------------------------------------------------------------
# preload_model / is_model_loaded
# ---------------------------------------------------------------------------


def test_preload_model_calls_get_model():
    with patch("app.services.embedder._get_model") as mock_get:
        preload_model()
        mock_get.assert_called_once()


def test_is_model_loaded_reflects_module_state(monkeypatch):
    monkeypatch.setattr(embedder_module, "_model", None)
    assert is_model_loaded() is False
    monkeypatch.setattr(embedder_module, "_model", object())
    assert is_model_loaded() is True
