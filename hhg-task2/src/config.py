"""
Centralized configuration via pydantic-settings.

All environment variables are loaded once at startup and validated.
Access config anywhere via: `from src.config import settings`
"""

from functools import lru_cache
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # --- App ---
    app_env: str = Field(default="development", description="Runtime environment")
    app_host: str = Field(default="0.0.0.0", description="Server bind host")
    app_port: int = Field(default=8000, description="Server bind port")
    max_latency_ms: int = Field(default=200, description="Target max latency in ms")
    log_level: str = Field(default="INFO", description="Logging level")
    answer_mode: str = Field(default="fast", description="Answer mode ('fast' extractive or 'generative')")

    # --- STT (ElevenLabs) ---
    elevenlabs_api_key: str = Field(default="", description="ElevenLabs API key")

    # --- LLM (Groq - Primary) ---
    groq_api_key: str = Field(default="", description="Groq API key")

    # --- LLM (OpenAI - Fallback) ---
    openai_api_key: str = Field(default="", description="OpenAI API key (fallback)")

    # --- Local vector store ---
    vector_store_type: str = Field(default="local", description="Active vector store type")

    # --- HuggingFace ---
    hf_token: str = Field(default="", description="HuggingFace access token")

    # --- Embedding ---
    embedding_model: str = Field(
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        description="Sentence transformer model name",
    )
    embedding_dimension: int = Field(default=384, description="Embedding vector dimension")

    # --- Retrieval ---
    retrieval_top_k: int = Field(default=5, description="Number of top results to retrieve")
    retrieval_namespace: str = Field(default="demo_fast", description="Active local retrieval namespace")

    # --- Guardrails ---
    off_topic_threshold: float = Field(
        default=0.10, description="Cosine similarity threshold below which a query is off-topic"
    )
    grounding_threshold: float = Field(
        default=0.58, description="Entailment/similarity threshold for grounding check"
    )
    grounding_threshold_gu: float = Field(
        default=0.40, description="Calibrated retrieval threshold for Gujarati evidence"
    )

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",
    }


@lru_cache()
def get_settings() -> Settings:
    """Return cached settings instance (singleton)."""
    return Settings()


# Convenience alias — import this directly
settings = get_settings()
