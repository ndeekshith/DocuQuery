"""Database management for document metadata."""

# Robust import: prefer stdlib sqlite3; if missing _sqlite3 binding, fall back to pysqlite3
try:
    import sqlite3  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - environment specific
    try:
        from pysqlite3 import dbapi2 as sqlite3  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "sqlite3 module is unavailable and pysqlite3 fallback failed."
        ) from exc
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any
from src.config import config
from src.logger import logger


class DocumentDatabase:
    """Manages document metadata in SQLite database."""
    
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or config.database.path
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    file_size INTEGER,
                    upload_date TEXT NOT NULL,
                    processed_date TEXT,
                    chunk_count INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'pending',
                    metadata TEXT,
                    preview TEXT
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS embeddings_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content_hash TEXT UNIQUE NOT NULL,
                    embedding BLOB NOT NULL,
                    created_date TEXT NOT NULL
                )
            """)
            conn.commit()
        logger.info(f"Database initialized at {self.db_path}")
    
    def add_document(self, filename: str, file_type: str, file_size: int, 
                     metadata: Optional[Dict] = None, preview: Optional[str] = None) -> int:
        """Add a new document to the database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO documents (filename, file_type, file_size, upload_date, metadata, preview)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                filename,
                file_type,
                file_size,
                datetime.now().isoformat(),
                json.dumps(metadata) if metadata else None,
                preview
            ))
            conn.commit()
            doc_id = cursor.lastrowid
        logger.info(f"Added document: {filename} (ID: {doc_id})")
        return doc_id
    
    def update_document_status(self, doc_id: int, status: str, chunk_count: Optional[int] = None):
        """Update document processing status."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            if chunk_count is not None:
                cursor.execute("""
                    UPDATE documents 
                    SET status = ?, processed_date = ?, chunk_count = ?
                    WHERE id = ?
                """, (status, datetime.now().isoformat(), chunk_count, doc_id))
            else:
                cursor.execute("""
                    UPDATE documents 
                    SET status = ?, processed_date = ?
                    WHERE id = ?
                """, (status, datetime.now().isoformat(), doc_id))
            conn.commit()
        logger.info(f"Updated document {doc_id} status to {status}")
    
    def get_document(self, doc_id: int) -> Optional[Dict[str, Any]]:
        """Get document by ID."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM documents WHERE id = ?", (doc_id,))
            row = cursor.fetchone()
            if row:
                doc = dict(row)
                if doc['metadata']:
                    doc['metadata'] = json.loads(doc['metadata'])
                return doc
        return None
    
    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Get all documents."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM documents ORDER BY upload_date DESC")
            rows = cursor.fetchall()
            documents = []
            for row in rows:
                doc = dict(row)
                if doc['metadata']:
                    doc['metadata'] = json.loads(doc['metadata'])
                documents.append(doc)
        return documents
    
    def delete_document(self, doc_id: int) -> bool:
        """Delete document by ID."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
            conn.commit()
            deleted = cursor.rowcount > 0
        if deleted:
            logger.info(f"Deleted document {doc_id}")
        return deleted
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) as total, SUM(chunk_count) as total_chunks, SUM(file_size) as total_size FROM documents")
            row = cursor.fetchone()
            cursor.execute("SELECT file_type, COUNT(*) as count FROM documents GROUP BY file_type")
            type_counts = {row[0]: row[1] for row in cursor.fetchall()}
        
        return {
            'total_documents': row[0] or 0,
            'total_chunks': row[1] or 0,
            'total_size_bytes': row[2] or 0,
            'documents_by_type': type_counts
        }
    
    def clear_all_documents(self):
        """Clear all documents from database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM documents")
            cursor.execute("DELETE FROM embeddings_cache")
            conn.commit()
        logger.info("Cleared all documents from database")
    
    def cache_embedding(self, content_hash: str, embedding: bytes):
        """Cache an embedding."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            try:
                cursor.execute("""
                    INSERT INTO embeddings_cache (content_hash, embedding, created_date)
                    VALUES (?, ?, ?)
                """, (content_hash, embedding, datetime.now().isoformat()))
                conn.commit()
            except sqlite3.IntegrityError:
                # Already exists
                pass
    
    def get_cached_embedding(self, content_hash: str) -> Optional[bytes]:
        """Get cached embedding."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT embedding FROM embeddings_cache WHERE content_hash = ?", (content_hash,))
            row = cursor.fetchone()
            return row[0] if row else None
