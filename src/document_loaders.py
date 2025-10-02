"""Document loaders for various file formats."""

import os
import csv
import asyncio
from pathlib import Path
from typing import List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
import hashlib

from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    TextLoader,
    UnstructuredHTMLLoader,
    UnstructuredRTFLoader
)
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from src.config import config
from src.logger import logger
from src.database import DocumentDatabase


class DocumentLoader:
    """Handles loading and processing of various document formats."""
    
    def __init__(self):
        self.config = config
        self.db = DocumentDatabase()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.document_processing.chunk_size,
            chunk_overlap=config.document_processing.chunk_overlap
        )
    
    def get_file_hash(self, file_path: str) -> str:
        """Generate hash for file content."""
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def get_preview(self, file_path: str, file_type: str) -> str:
        """Generate preview of document content."""
        try:
            preview_length = config.document_processing.preview_length
            
            if file_type == 'txt' or file_type == 'md':
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read(preview_length)
                    return content
            
            elif file_type == 'csv':
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    reader = csv.reader(f)
                    lines = []
                    for i, row in enumerate(reader):
                        if i >= 5:  # First 5 rows
                            break
                        lines.append(', '.join(row))
                    return '\n'.join(lines)
            
            elif file_type == 'pdf':
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                if docs:
                    return docs[0].page_content[:preview_length]
            
            elif file_type == 'docx':
                loader = UnstructuredWordDocumentLoader(file_path)
                docs = loader.load()
                if docs:
                    return docs[0].page_content[:preview_length]
            
            elif file_type == 'html':
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read(preview_length * 2)  # HTML has tags
                    return content[:preview_length]
            
            elif file_type == 'rtf':
                loader = UnstructuredRTFLoader(file_path)
                docs = loader.load()
                if docs:
                    return docs[0].page_content[:preview_length]
            
            return "Preview not available"
        
        except Exception as e:
            logger.error(f"Error generating preview for {file_path}: {e}")
            return "Preview generation failed"
    
    def load_single_document(self, file_path: str, filename: str) -> Tuple[List[Document], Optional[int]]:
        """Load a single document and return chunks."""
        file_type = Path(filename).suffix.lower().lstrip('.')
        
        try:
            logger.info(f"Loading document: {filename} (type: {file_type})")
            
            # Generate preview
            preview = self.get_preview(file_path, file_type) if config.document_processing.enable_preview else None
            
            # Add to database
            file_size = os.path.getsize(file_path)
            doc_id = self.db.add_document(
                filename=filename,
                file_type=file_type,
                file_size=file_size,
                preview=preview
            )
            
            # Load document based on type
            documents = []
            
            if file_type == 'pdf':
                loader = PyPDFLoader(file_path)
                documents = loader.load()
            
            elif file_type == 'docx':
                loader = UnstructuredWordDocumentLoader(file_path)
                documents = loader.load()
            
            elif file_type in ['txt', 'md']:
                loader = TextLoader(file_path, encoding='utf-8')
                documents = loader.load()
            
            elif file_type == 'csv':
                documents = self._load_csv(file_path)
            
            elif file_type == 'html':
                loader = UnstructuredHTMLLoader(file_path)
                documents = loader.load()
            
            elif file_type == 'rtf':
                loader = UnstructuredRTFLoader(file_path)
                documents = loader.load()
            
            else:
                logger.warning(f"Unsupported file type: {file_type}")
                self.db.update_document_status(doc_id, 'failed')
                return [], None
            
            # Add metadata
            for doc in documents:
                doc.metadata['source'] = filename
                doc.metadata['file_type'] = file_type
                doc.metadata['doc_id'] = doc_id
            
            # Split into chunks
            chunks = self.text_splitter.split_documents(documents)
            
            # Update database
            self.db.update_document_status(doc_id, 'processed', len(chunks))
            
            logger.info(f"Successfully loaded {filename}: {len(chunks)} chunks")
            return chunks, doc_id
        
        except Exception as e:
            logger.error(f"Error loading {filename}: {e}")
            if 'doc_id' in locals():
                self.db.update_document_status(doc_id, 'failed')
            return [], None
    
    def _load_csv(self, file_path: str) -> List[Document]:
        """Load CSV file as documents."""
        documents = []
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    content = '\n'.join([f"{k}: {v}" for k, v in row.items()])
                    doc = Document(
                        page_content=content,
                        metadata={'row': i, 'source': file_path}
                    )
                    documents.append(doc)
        except Exception as e:
            logger.error(f"Error loading CSV {file_path}: {e}")
        
        return documents
    
    async def load_documents_async(self, uploaded_files) -> List[Document]:
        """Load multiple documents asynchronously."""
        temp_dir = Path(config.document_processing.temp_directory)
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        all_chunks = []
        
        # Save files to temp directory
        file_paths = []
        for uploaded_file in uploaded_files:
            file_path = temp_dir / uploaded_file.name
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            file_paths.append((str(file_path), uploaded_file.name))
        
        # Process files in parallel
        with ThreadPoolExecutor(max_workers=config.performance.max_workers) as executor:
            loop = asyncio.get_event_loop()
            tasks = [
                loop.run_in_executor(executor, self.load_single_document, file_path, filename)
                for file_path, filename in file_paths
            ]
            results = await asyncio.gather(*tasks)
        
        # Collect chunks
        for chunks, doc_id in results:
            all_chunks.extend(chunks)
        
        # Cleanup temp files
        for file_path, _ in file_paths:
            try:
                os.remove(file_path)
            except Exception as e:
                logger.warning(f"Could not remove temp file {file_path}: {e}")
        
        return all_chunks
    
    def load_documents_sync(self, uploaded_files) -> List[Document]:
        """Load multiple documents synchronously with progress tracking."""
        temp_dir = Path(config.document_processing.temp_directory)
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        all_chunks = []
        
        for uploaded_file in uploaded_files:
            file_path = temp_dir / uploaded_file.name
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            chunks, doc_id = self.load_single_document(str(file_path), uploaded_file.name)
            all_chunks.extend(chunks)
            
            # Cleanup
            try:
                os.remove(file_path)
            except Exception as e:
                logger.warning(f"Could not remove temp file {file_path}: {e}")
        
        return all_chunks
