"""
Central configuration loaded from the .env file.
All modules import `settings` from here — never read os.environ directly.
"""

from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )

    # Authentication
    MNEMONIC_API_TOKEN: str = Field(
        description="Static bearer token. Generate with: openssl rand -hex 32"
    )

    #  PostgreSQL 
    POSTGRES_URL: str = Field(
        description="Async SQLAlchemy DSN: postgresql+asyncpg://user:pass@host:port/db"
    )
    POSTGRES_SYNC_URL: str = Field(
        description="Sync DSN for Alembic migrations."
    )

    #  Qdrant 
    QDRANT_HOST: str = Field(default="localhost")
    QDRANT_PORT: int = Field(default=6333)
    QDRANT_API_KEY: str = Field(default="")
    QDRANT_COLLECTION_NAME: str = Field(default="knowledge_base")
    QDRANT_VECTOR_SIZE: int = Field(default=1024)

    # Neo4j 
    NEO4J_URI: str = Field(default="bolt://localhost:7687")
    NEO4J_USER: str = Field(default="neo4j")
    NEO4J_PASSWORD: str

    # Redis 
    REDIS_URL: str = Field(default="redis://localhost:6379/0")

    #  Secondary Node 
    SECONDARY_NODE_URL: str = Field(
        description="Base URL of the BGE-M3 + NLI service on the GTX 1650 machine."
    )
    SECONDARY_NODE_TIMEOUT: float = Field(default=10.0)

    # Ollama (local LLM for classification and agents) 
    OLLAMA_URL: str = Field(
        default="http://ollama:11434",
        description="Base URL of the Ollama service inside Docker network."
    )
    OLLAMA_MODEL: str = Field(
        default="qwen2.5-coder:7b",
        description="Ollama model tag for classification and agent tasks."
    )

    PISTON_URL: str = Field(default="http://piston:2000")

    # Application 
    LOG_LEVEL: str = Field(default="INFO")
    ENVIRONMENT: str = Field(default="production")

    # KV cache safety threshold — warn if any real-time prompt exceeds this.
    PROMPT_TOKEN_WARN_THRESHOLD: int = Field(
        default=2500,
        description="Log a warning if a real-time prompt exceeds this token count."
    )

    # Hard retrieval caps
    RETRIEVAL_TOP_K_REALTIME: int = Field(default=12)
    RETRIEVAL_TOP_K_ASYNC: int = Field(default=12)
    RETRIEVAL_MAX_CHUNKS_PER_SOURCE: int = Field(default=2)

@lru_cache
def get_settings() -> Settings:
    return Settings()


settings: Settings = get_settings()