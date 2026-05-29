from concurrent.futures import ThreadPoolExecutor

from app.services.embedder import embed

_EMBEDDING_DIM = 768


async def test_embed_returns_list_of_lists():
    result = await embed(["hello world"])
    assert isinstance(result, list)
    assert isinstance(result[0], list)


async def test_embed_correct_dimension():
    result = await embed(["test sentence"])
    assert len(result[0]) == _EMBEDDING_DIM


async def test_embed_multiple_texts():
    texts = ["first sentence", "second sentence", "third sentence"]
    result = await embed(texts)
    assert len(result) == 3
    assert all(len(v) == _EMBEDDING_DIM for v in result)


async def test_embed_values_are_floats():
    result = await embed(["sample text"])
    assert all(isinstance(v, float) for v in result[0])


async def test_embed_different_texts_produce_different_vectors():
    a = await embed(["the dog sat on the mat"])
    b = await embed(["quantum entanglement in physics"])
    assert a[0] != b[0]


async def test_embed_with_explicit_executor():
    """embed() accepts a dedicated ThreadPoolExecutor and routes work through it."""
    with ThreadPoolExecutor(max_workers=1) as executor:
        result = await embed(["hello world"], executor)
    assert isinstance(result, list)
    assert isinstance(result[0], list)
    assert len(result[0]) == _EMBEDDING_DIM


async def test_embed_similar_texts_are_closer():
    """Cosine distance between similar sentences < distance to unrelated one."""
    import math

    def dot(u: list[float], v: list[float]) -> float:
        return sum(x * y for x, y in zip(u, v))

    def norm(u: list[float]) -> float:
        return math.sqrt(sum(x**2 for x in u))

    def cosine_sim(u: list[float], v: list[float]) -> float:
        return dot(u, v) / (norm(u) * norm(v))

    a, b, c = await embed(
        [
            "Take this medication with food.",
            "This drug should be taken with meals.",
            "The stock market crashed yesterday.",
        ]
    )
    assert cosine_sim(a, b) > cosine_sim(a, c)
