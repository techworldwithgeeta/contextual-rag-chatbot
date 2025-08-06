"""
Configuration settings for the Contextual RAG Chatbot.
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Application settings."""
    
    # OpenAI Configuration
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
    EMBEDDING_DIMENSION: int = int(os.getenv("EMBEDDING_DIMENSION", "1536"))
    
    # Ollama Configuration
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama2:7b")
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    # Database Configuration
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5433/vectordb")
    
    # Vector Store Configuration
    VECTOR_STORE_TOP_K: int = int(os.getenv("VECTOR_STORE_TOP_K", "5"))
    
    # RAG Configuration
    #USE_CREW_AI_ENHANCEMENT: bool = os.getenv("USE_CREW_AI_ENHANCEMENT", "false").lower() == "true"
    USE_CREW_AI_ENHANCEMENT: bool = os.getenv("USE_CREW_AI_ENHANCEMENT", "true").lower() == "true"
    
    # Web Interface Configuration
    WEB_HOST: str = os.getenv("WEB_HOST", "localhost")
    WEB_PORT: int = int(os.getenv("WEB_PORT", "8001"))
    
    # Logging Configuration
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: str = os.getenv("LOG_FILE", "logs/rag_chatbot.log")
    
    # Phoenix Configuration
    PHOENIX_PORT: int = int(os.getenv("PHOENIX_PORT", "6006"))
    
    # Open WebUI Configuration
    OPEN_WEBUI_URL: str = os.getenv("OPEN_WEBUI_URL", "http://localhost:3000")
    
    # Document Processing Configuration
    FAST_DOCUMENT_PROCESSING: bool = os.getenv("FAST_DOCUMENT_PROCESSING", "true").lower() == "true"
    
    # CSV Generation Configuration
    AUTO_GENERATE_CSV_AFTER_EVALUATION: bool = os.getenv("AUTO_GENERATE_CSV_AFTER_EVALUATION", "true").lower() == "true"
    SIMPLIFIED_CSV_EXPORT: bool = os.getenv("SIMPLIFIED_CSV_EXPORT", "true").lower() == "true"
    
    class Config:
        # Don't load .env file to avoid conflicts
        # env_file = ".env"
        extra = "ignore"  # Ignore extra fields from environment

# Create global settings instance
settings = Settings()

# Force OpenAI embedding model (override .env file)
if os.getenv("EMBEDDING_MODEL"):
    settings.EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
if os.getenv("EMBEDDING_DIMENSION"):
    settings.EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION")) 