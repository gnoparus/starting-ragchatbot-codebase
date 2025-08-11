import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass
class Config:
    """Configuration settings for the RAG system"""

    # Anthropic API settings
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    ANTHROPIC_MODEL: str = "claude-sonnet-4-20250514"

    # Embedding model settings
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    # Document processing settings
    CHUNK_SIZE: int = 800  # Size of text chunks for vector storage
    CHUNK_OVERLAP: int = 100  # Characters to overlap between chunks
    MAX_RESULTS: int = 5  # Maximum search results to return
    MAX_HISTORY: int = 2  # Number of conversation messages to remember
    MAX_TOOL_ROUNDS: int = 2  # Maximum sequential tool calling rounds

    # Database paths
    CHROMA_PATH: str = "./chroma_db"  # ChromaDB storage location

    # Logging configuration
    LOG_LEVEL: str = os.getenv(
        "LOG_LEVEL", "INFO"
    )  # TRACE, DEBUG, INFO, SUCCESS, WARNING, ERROR, CRITICAL
    LOG_TO_CONSOLE: bool = os.getenv("LOG_TO_CONSOLE", "true").lower() == "true"
    LOG_TO_FILE: bool = os.getenv("LOG_TO_FILE", "true").lower() == "true"
    LOG_FILE_PATH: str = os.getenv("LOG_FILE_PATH", "./logs")
    LOG_ROTATION: str = os.getenv("LOG_ROTATION", "10 MB")  # File size for rotation
    LOG_RETENTION: str = os.getenv("LOG_RETENTION", "7 days")  # How long to keep logs


config = Config()
