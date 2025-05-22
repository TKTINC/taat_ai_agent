"""
Configuration for memory systems.

This module extends the base configuration with settings for memory systems.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Literal

from src.agent_core.config import AgentConfig


@dataclass
class VectorDBConfig:
    """Configuration for vector database."""
    provider: Literal["pinecone"] = "pinecone"
    api_key: str = ""
    index_name: str = "taat-episodic-memory"
    dimension: int = 1536  # OpenAI embedding dimension
    metric: str = "cosine"
    namespace: Optional[str] = None
    environment: Optional[str] = None  # For Pinecone


@dataclass
class EmbeddingConfig:
    """Configuration for embedding model."""
    provider: Literal["openai"] = "openai"
    api_key: str = ""
    model: str = "text-embedding-ada-002"
    dimension: int = 1536
    batch_size: int = 8


@dataclass
class DatabaseConfig:
    """Configuration for database."""
    provider: Literal["sqlite", "postgresql"] = "sqlite"
    connection_string: str = "sqlite:///memory.db"
    pool_size: int = 5
    max_overflow: int = 10
    pool_timeout: int = 30
    pool_recycle: int = 1800


@dataclass
class MemoryConfig:
    """Configuration for memory systems."""
    vector_db: VectorDBConfig = field(default_factory=VectorDBConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    max_episodic_memories: int = 5
    max_procedural_patterns: int = 3
    memory_refresh_interval: int = 3600  # seconds


@dataclass
class EnhancedAgentConfig(AgentConfig):
    """Enhanced configuration for the TAAT Agent with memory systems."""
    memory: MemoryConfig = field(default_factory=MemoryConfig)


def load_enhanced_config() -> EnhancedAgentConfig:
    """
    Load enhanced configuration from environment variables.
    
    Returns:
        EnhancedAgentConfig: The loaded configuration
        
    Raises:
        ValueError: If required environment variables are missing
    """
    # Load base config
    base_config = super(EnhancedAgentConfig, EnhancedAgentConfig).load_config()
    
    # Vector DB config
    vector_db_config = VectorDBConfig(
        provider="pinecone",  # Only supporting Pinecone for now
        api_key=os.environ.get("PINECONE_API_KEY", ""),
        index_name=os.environ.get("PINECONE_INDEX_NAME", "taat-episodic-memory"),
        dimension=int(os.environ.get("VECTOR_DIMENSION", "1536")),
        metric=os.environ.get("VECTOR_METRIC", "cosine"),
        environment=os.environ.get("PINECONE_ENVIRONMENT", None)
    )
    
    # Embedding config
    embedding_config = EmbeddingConfig(
        provider="openai",  # Only supporting OpenAI for now
        api_key=os.environ.get("OPENAI_API_KEY", ""),
        model=os.environ.get("EMBEDDING_MODEL", "text-embedding-ada-002"),
        dimension=int(os.environ.get("EMBEDDING_DIMENSION", "1536")),
        batch_size=int(os.environ.get("EMBEDDING_BATCH_SIZE", "8"))
    )
    
    # Database config
    db_provider = os.environ.get("DB_PROVIDER", "sqlite").lower()
    if db_provider not in ["sqlite", "postgresql"]:
        raise ValueError(f"Unsupported database provider: {db_provider}")
    
    connection_string = ""
    if db_provider == "sqlite":
        connection_string = os.environ.get("SQLITE_CONNECTION", "sqlite:///memory.db")
    else:  # postgresql
        pg_user = os.environ.get("POSTGRES_USER", "")
        pg_password = os.environ.get("POSTGRES_PASSWORD", "")
        pg_host = os.environ.get("POSTGRES_HOST", "localhost")
        pg_port = os.environ.get("POSTGRES_PORT", "5432")
        pg_db = os.environ.get("POSTGRES_DB", "taat")
        connection_string = f"postgresql://{pg_user}:{pg_password}@{pg_host}:{pg_port}/{pg_db}"
    
    database_config = DatabaseConfig(
        provider=db_provider,
        connection_string=connection_string,
        pool_size=int(os.environ.get("DB_POOL_SIZE", "5")),
        max_overflow=int(os.environ.get("DB_MAX_OVERFLOW", "10")),
        pool_timeout=int(os.environ.get("DB_POOL_TIMEOUT", "30")),
        pool_recycle=int(os.environ.get("DB_POOL_RECYCLE", "1800"))
    )
    
    # Memory config
    memory_config = MemoryConfig(
        vector_db=vector_db_config,
        embedding=embedding_config,
        database=database_config,
        max_episodic_memories=int(os.environ.get("MAX_EPISODIC_MEMORIES", "5")),
        max_procedural_patterns=int(os.environ.get("MAX_PROCEDURAL_PATTERNS", "3")),
        memory_refresh_interval=int(os.environ.get("MEMORY_REFRESH_INTERVAL", "3600"))
    )
    
    # Create enhanced config
    return EnhancedAgentConfig(
        llm_settings=base_config.llm_settings,
        debug_mode=base_config.debug_mode,
        log_level=base_config.log_level,
        max_history=base_config.max_history,
        memory=memory_config
    )
