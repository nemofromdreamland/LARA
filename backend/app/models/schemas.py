from typing import Annotated, Literal

from pydantic import BaseModel, Field


class PrescriptionEntry(BaseModel):
    """One medication line from a parsed prescription."""

    drug_name: str
    dosage: str | None = None
    frequency: str | None = None
    duration: str | None = None
    instructions: str | None = None


class SessionResponse(BaseModel):
    session_id: str


class ComponentHealth(BaseModel):
    status: Literal["ok", "degraded", "unavailable"]
    detail: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    components: dict[str, "ComponentHealth"]


class UploadResponse(BaseModel):
    session_id: str
    drugs_found: list[str]
    missing_leaflets: list[str]
    status: Literal["ok", "no_leaflets_found"]


class UploadJobResponse(BaseModel):
    job_id: str
    session_id: str
    status: Literal["processing"]


class JobStatusResponse(BaseModel):
    job_id: str
    session_id: str
    status: Literal["processing", "done", "failed"]
    drugs_found: list[str] = []
    missing_leaflets: list[str] = []
    error: str | None = None


class Source(BaseModel):
    drug_name: str
    section: str


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=8000)


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=36, max_length=36)
    question: str = Field(min_length=2, max_length=4000)
    history: Annotated[list[ChatTurn], Field(max_length=10)] = []


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]


class InteractionsRequest(BaseModel):
    session_id: str


class InteractionFlag(BaseModel):
    drug_a: str  # drug whose leaflet mentions the interaction
    drug_b: str  # drug being mentioned
    excerpt: str  # supporting text passage from the leaflet


class InteractionsResponse(BaseModel):
    session_id: str
    pairs_checked: int
    interactions: list[InteractionFlag]
