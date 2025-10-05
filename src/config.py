"""Configuration management for DocuQuery."""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional


class Config:
    """Configuration manager that loads settings from config.yaml."""
    
    def __init__(self, config_path: Optional[str] = "config.yaml"):
        proposed_path = Path(config_path) if config_path else Path("config.yaml")
        if proposed_path.is_absolute() and proposed_path.exists():
            self.config_path = proposed_path
        else:
            module_root = Path(__file__).resolve().parents[1]  # points to DocuQuery/
            root_candidate = module_root / "config.yaml"
            if root_candidate.exists():
                self.config_path = root_candidate
            elif proposed_path.exists():
                self.config_path = proposed_path
            else:
                # As a last resort, try CWD
                cwd_candidate = Path.cwd() / "config.yaml"
                if cwd_candidate.exists():
                    self.config_path = cwd_candidate
                else:
            
                    self.config_path = proposed_path
        
        self._config: Dict[str, Any] = {}
        self.load_config()
        self._ensure_directories()
    
    def load_config(self):
        """Load configuration from YAML file."""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                self._config = yaml.safe_load(f) or {}
        else:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
    
    def _ensure_directories(self):
        """Create necessary directories if they don't exist."""
        directories = [
            self.vectorstore.persist_directory,
            self.document_processing.temp_directory,
            self.performance.embedding_cache_dir,
            Path(self.logging.file).parent,
            Path(self.database.path).parent,
            self.embeddings.cache_folder,
        ]
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def get(self, key: str, default=None):
        """Get configuration value by dot-notation key."""
        keys = key.split('.')
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        return value
    
    @property
    def ollama(self):
        """Ollama configuration."""
        return type('OllamaConfig', (), self._config.get('ollama', {}))()
    
    @property
    def embeddings(self):
        """Embeddings configuration."""
        return type('EmbeddingsConfig', (), self._config.get('embeddings', {}))()
    
    @property
    def vectorstore(self):
        """Vector store configuration."""
        return type('VectorStoreConfig', (), self._config.get('vectorstore', {}))()
    
    @property
    def document_processing(self):
        """Document processing configuration."""
        return type('DocumentProcessingConfig', (), self._config.get('document_processing', {}))()
    
    @property
    def search(self):
        """Search configuration."""
        return type('SearchConfig', (), self._config.get('search', {}))()
    
    @property
    def ui(self):
        """UI configuration."""
        return type('UIConfig', (), self._config.get('ui', {}))()
    
    @property
    def performance(self):
        """Performance configuration."""
        return type('PerformanceConfig', (), self._config.get('performance', {}))()
    
    @property
    def logging(self):
        """Logging configuration."""
        return type('LoggingConfig', (), self._config.get('logging', {}))()
    
    @property
    def database(self):
        """Database configuration."""
        return type('DatabaseConfig', (), self._config.get('database', {}))()



config = Config()
