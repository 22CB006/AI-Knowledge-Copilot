"""
Configuration management for the application.
"""
import os
import yaml
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Application configuration."""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self._config: Dict[str, Any] = {}
        self._load_config()
    
    def _load_config(self):
        """Load configuration from YAML file."""
        config_file = Path(self.config_path)
        if config_file.exists():
            with open(config_file, 'r') as f:
                self._config = yaml.safe_load(f)
        else:
            # Use defaults if config file doesn't exist
            self._config = self._get_defaults()
    
    def _get_defaults(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "embedding": {
                "model": "all-MiniLM-L6-v2",
                "dimension": 384
            },
            "chunking": {
                "chunk_size": 512,
                "overlap": 50
            },
            "retrieval": {
                "top_k": 10,
                "similarity_threshold": 0.5
            },
            "reranking": {
                "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
                "top_n": 5
            },
            "llm": {
                "provider": "openai",
                "temperature": 0.7,
                "max_tokens": 500
            },
            "storage": {
                "faiss_index_path": "./data/faiss_index",
                "metadata_db_path": "./data/metadata.db"
            },
            "api": {
                "host": "0.0.0.0",
                "port": 8000,
                "cors_origins": ["http://localhost:3000"]
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key (supports nested keys with dot notation)."""
        keys = key.split('.')
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
        return value if value is not None else default
    
    @property
    def embedding_model(self) -> str:
        return self.get("embedding.model", "all-MiniLM-L6-v2")
    
    @property
    def chunk_size(self) -> int:
        return self.get("chunking.chunk_size", 512)
    
    @property
    def chunk_overlap(self) -> int:
        return self.get("chunking.overlap", 50)
    
    @property
    def top_k(self) -> int:
        return self.get("retrieval.top_k", 10)
    
    @property
    def top_n_rerank(self) -> int:
        return self.get("reranking.top_n", 5)
    
    @property
    def faiss_index_path(self) -> str:
        return self.get("storage.faiss_index_path", "./data/faiss_index")
    
    @property
    def metadata_db_path(self) -> str:
        return self.get("storage.metadata_db_path", "./data/metadata.db")
    
    @property
    def openai_api_key(self) -> str:
        return os.getenv("OPENAI_API_KEY", "")
    
    @property
    def encryption_key(self) -> str:
        return os.getenv("ENCRYPTION_KEY", "")
    
    @property
    def cors_origins(self) -> list:
        env_origins = os.getenv("CORS_ORIGINS", "")
        if env_origins:
            return [origin.strip() for origin in env_origins.split(",")]
        return self.get("api.cors_origins", ["http://localhost:3000"])


# Global config instance
config = Config()
