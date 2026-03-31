import httpx
from groq import AsyncGroq

from app.config import LLMProvider, settings

SYSTEM_PROMPT = (
    "You are LARA, a medical information assistant. "
    "Answer ONLY using the context provided below. "
    "If the context does not contain the answer, respond: "
    "'This information is not available in the provided leaflets.' "
    "Never add information from general knowledge. "
    "Always cite the section (e.g. 'According to the Warnings section...')."
)

_MODEL_GROQ = "llama-3.3-70b-versatile"
_MODEL_CEREBRAS = "llama3.3-70b"


def _build_prompt(context: str, question: str) -> str:
    return f"Context:\n{context}\n\nQuestion: {question}"


async def generate(context: str, question: str) -> str:
    """Send context + question to the configured LLM and return the answer.

    Uses Groq by default; switches to Cerebras when LLM_PROVIDER=cerebras.
    """
    prompt = _build_prompt(context, question)

    if settings.llm_provider == LLMProvider.cerebras:
        return await _call_cerebras(prompt)
    return await _call_groq(prompt)


async def _call_groq(prompt: str) -> str:
    client = AsyncGroq(api_key=settings.groq_api_key)
    response = await client.chat.completions.create(
        model=_MODEL_GROQ,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )
    return response.choices[0].message.content


async def _call_cerebras(prompt: str) -> str:
    # Cerebras uses an OpenAI-compatible endpoint
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            "https://api.cerebras.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {settings.cerebras_api_key}"},
            json={
                "model": _MODEL_CEREBRAS,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.0,
            },
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
