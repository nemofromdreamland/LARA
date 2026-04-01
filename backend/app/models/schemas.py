from pydantic import BaseModel


class SessionResponse(BaseModel):
    session_id: str


class HealthResponse(BaseModel):
    status: str


class UploadResponse(BaseModel):
    session_id: str
    drugs_found: list[str]
    missing_leaflets: list[str]
    status: str


class Source(BaseModel):
    drug_name: str
    section: str


class ChatRequest(BaseModel):
    session_id: str
    question: str


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
