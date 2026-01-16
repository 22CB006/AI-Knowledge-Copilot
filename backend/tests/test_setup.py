"""
Basic tests to verify project setup.
"""
import pytest
from pathlib import Path


def test_project_structure():
    """Test that basic project structure exists."""
    backend_dir = Path(__file__).parent.parent
    
    # Check key directories exist
    assert (backend_dir / "src").exists()
    assert (backend_dir / "tests").exists()
    assert (backend_dir / "data").exists()
    
    # Check key files exist
    assert (backend_dir / "requirements.txt").exists()
    assert (backend_dir / "config.yaml").exists()
    assert (backend_dir / "src" / "__init__.py").exists()
    assert (backend_dir / "src" / "main.py").exists()
    assert (backend_dir / "src" / "config.py").exists()


def test_config_loading():
    """Test that configuration can be loaded."""
    from src.config import Config
    
    config = Config()
    
    # Test default values
    assert config.embedding_model == "all-MiniLM-L6-v2"
    assert config.chunk_size == 512
    assert config.chunk_overlap == 50
    assert config.top_k == 10
    assert config.top_n_rerank == 5


def test_fastapi_app_creation():
    """Test that FastAPI app can be created."""
    from src.main import app
    
    assert app is not None
    assert app.title == "AI Knowledge Copilot API"
    assert app.version == "1.0.0"
