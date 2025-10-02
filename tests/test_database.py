"""Tests for database module."""

import pytest
import tempfile
import os
from src.database import DocumentDatabase


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as f:
        db_path = f.name
    
    db = DocumentDatabase(db_path)
    yield db
    
    # Cleanup
    if os.path.exists(db_path):
        os.remove(db_path)


def test_add_document(temp_db):
    """Test adding a document."""
    doc_id = temp_db.add_document(
        filename="test.pdf",
        file_type="pdf",
        file_size=1024,
        metadata={"key": "value"}
    )
    
    assert doc_id > 0
    
    doc = temp_db.get_document(doc_id)
    assert doc is not None
    assert doc['filename'] == "test.pdf"
    assert doc['file_type'] == "pdf"


def test_update_document_status(temp_db):
    """Test updating document status."""
    doc_id = temp_db.add_document(
        filename="test.pdf",
        file_type="pdf",
        file_size=1024
    )
    
    temp_db.update_document_status(doc_id, 'processed', 10)
    
    doc = temp_db.get_document(doc_id)
    assert doc['status'] == 'processed'
    assert doc['chunk_count'] == 10


def test_get_statistics(temp_db):
    """Test getting statistics."""
    temp_db.add_document("test1.pdf", "pdf", 1024)
    temp_db.add_document("test2.txt", "txt", 512)
    
    stats = temp_db.get_statistics()
    
    assert stats['total_documents'] == 2
    assert stats['total_size_bytes'] == 1536
    assert 'pdf' in stats['documents_by_type']
    assert 'txt' in stats['documents_by_type']


def test_delete_document(temp_db):
    """Test deleting a document."""
    doc_id = temp_db.add_document("test.pdf", "pdf", 1024)
    
    deleted = temp_db.delete_document(doc_id)
    assert deleted is True
    
    doc = temp_db.get_document(doc_id)
    assert doc is None
