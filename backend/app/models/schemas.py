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
