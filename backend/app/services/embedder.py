from sentence_transformers import SentenceTransformer

_MODEL_NAME = "all-MiniLM-L6-v2"
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def embed(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts using all-MiniLM-L6-v2 (local, no API key).

    Returns a list of 384-dimensional float vectors.
    """
    model = _get_model()
    return model.encode(texts, show_progress_bar=False).tolist()
