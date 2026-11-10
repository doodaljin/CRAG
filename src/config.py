from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Google Gemini API
    GOOGLE_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-3.5-flash"

    # Local NLP Models
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    RERANKER_MODEL: str = "ms-marco-TinyBERT-L-2-v2"

    # Project Directories & Storage
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    STORAGE_DIR: Path = BASE_DIR / "storage"
    QDRANT_PATH: Path = STORAGE_DIR / "qdrant"
    BM25_PATH: Path = STORAGE_DIR / "bm25.pkl"
    CHUNKS_PATH: Path = STORAGE_DIR / "chunks.pkl"

    # Vector & Lexical Search Configuration
    # Defaults to local disk storage at QDRANT_PATH when None or empty.
    # Set to ':memory:' for transient tests or provide a remote Qdrant URL.
    QDRANT_LOCATION: str | None = None
    QDRANT_COLLECTION: str = "crag_gemini_docs"
    TOP_K_RETRIEVAL: int = 10
    TOP_K_RERANK: int = 3
    MAX_CHUNK_TOKENS: int = 384

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()


def check_index_exists(storage_dir: Path | str | None = None) -> bool:
    """Checks whether the persistent offline index files exist without importing heavy ML packages."""
    dir_path = Path(storage_dir) if storage_dir else settings.STORAGE_DIR
    bm25_file = dir_path / "bm25.pkl"
    chunks_file = dir_path / "chunks.pkl"
    qdrant_dir = dir_path / "qdrant"
    return bm25_file.exists() and chunks_file.exists() and qdrant_dir.exists()
