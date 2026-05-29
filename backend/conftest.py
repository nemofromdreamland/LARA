import os

# Set required env vars before any app module is imported by test files.
# pydantic_settings reads os.environ at Settings() instantiation time, so
# these must be in place before `from app.main import app` fires in tests/.
os.environ.setdefault("LARA_API_KEY", "test-api-key")
os.environ.setdefault("GROQ_API_KEY", "test-groq-key")
