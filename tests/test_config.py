"""Tests for configuration module."""

import pytest
from pathlib import Path
from src.config import Config


def test_config_loads():
    """Test that config loads successfully."""
    config = Config()
    assert config is not None
    assert config.ollama.default_model is not None


def test_config_properties():
    """Test config properties."""
    config = Config()
    
    assert hasattr(config, 'ollama')
    assert hasattr(config, 'embeddings')
    assert hasattr(config, 'vectorstore')
    assert hasattr(config, 'document_processing')
    assert hasattr(config, 'search')
    assert hasattr(config, 'ui')


def test_config_get_method():
    """Test config get method."""
    config = Config()
    
    model = config.get('ollama.default_model')
    assert model is not None
    
    invalid = config.get('invalid.key', 'default')
    assert invalid == 'default'
