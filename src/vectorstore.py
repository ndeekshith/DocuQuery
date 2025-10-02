"""Vector store management with FAISS and hybrid search."""

import os
import pickle
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.retrievers import BM25Retriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from src.config import config
from src.logger import logger


class VectorStoreManager:
    """Manages FAISS vector store with hybrid search and reranking."""
    
    def __init__(self):
        self.config = config
        self.embeddings = self._initialize_embeddings()
        self.vectorstore: Optional[FAISS] = None
        self.bm25_retriever: Optional[BM25Retriever] = None
        self.reranker = None
        
        if config.search.use_reranker:
            self._initialize_reranker()
        
        # Try to load existing vectorstore
        self.load_vectorstore()
    
    def _initialize_embeddings(self) -> HuggingFaceEmbeddings:
        """Initialize embedding model."""
        logger.info(f"Initializing embeddings: {config.embeddings.model_name}")
        
        # Force CPU usage to avoid CUDA errors
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        
        return HuggingFaceEmbeddings(
            model_name=config.embeddings.model_name,
            cache_folder=config.embeddings.cache_folder,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'device': 'cpu', 'batch_size': 32}
        )
    
    def _initialize_reranker(self):
        """Initialize reranker model."""
        try:
            logger.info(f"Initializing reranker: {config.search.reranker_model}")
            model = HuggingFaceCrossEncoder(model_name=config.search.reranker_model)
            self.reranker = CrossEncoderReranker(model=model, top_n=config.search.top_k_rerank)
        except Exception as e:
            logger.error(f"Failed to initialize reranker: {e}")
            self.reranker = None
    
    def create_vectorstore(self, documents: List[Document]) -> bool:
        """Create or update vector store with documents."""
        if not documents:
            logger.warning("No documents provided to create vectorstore")
            return False
        
        try:
            persist_dir = Path(config.vectorstore.persist_directory)
            
            if self.vectorstore is not None:
                # Add to existing vectorstore
                logger.info(f"Adding {len(documents)} documents to existing vectorstore")
                self.vectorstore.add_documents(documents)
            else:
                # Create new vectorstore
                logger.info(f"Creating new vectorstore with {len(documents)} documents")
                
                if config.vectorstore.use_ivf_index and len(documents) > 1000:
                    # Use IVF index for large datasets
                    logger.info("Using IVF index for large dataset")
                    self.vectorstore = FAISS.from_documents(
                        documents,
                        self.embeddings
                    )
                    # Convert to IVF index
                    self._convert_to_ivf_index()
                else:
                    self.vectorstore = FAISS.from_documents(
                        documents,
                        self.embeddings
                    )
            
            # Save vectorstore
            self.save_vectorstore()
            
            # Update BM25 retriever for hybrid search
            if config.search.hybrid_search:
                self._update_bm25_retriever()
            
            logger.info("Vectorstore created/updated successfully")
            return True
        
        except Exception as e:
            logger.error(f"Error creating vectorstore: {e}")
            return False
    
    def _convert_to_ivf_index(self):
        """Convert flat index to IVF index for better performance."""
        try:
            import faiss
            
            # Get the index
            index = self.vectorstore.index
            d = index.d  # dimension
            
            # Create IVF index
            quantizer = faiss.IndexFlatL2(d)
            nlist = config.vectorstore.ivf_nlist
            new_index = faiss.IndexIVFFlat(quantizer, d, nlist)
            
            # Train and add vectors
            vectors = index.reconstruct_n(0, index.ntotal)
            new_index.train(vectors)
            new_index.add(vectors)
            
            # Replace index
            self.vectorstore.index = new_index
            logger.info(f"Converted to IVF index with {nlist} clusters")
        
        except Exception as e:
            logger.error(f"Failed to convert to IVF index: {e}")
    
    def _update_bm25_retriever(self):
        """Update BM25 retriever with current documents."""
        try:
            if self.vectorstore is None:
                return
            
            # Get all documents from vectorstore
            docs = self._get_all_documents()
            
            if docs:
                self.bm25_retriever = BM25Retriever.from_documents(docs)
                self.bm25_retriever.k = config.search.top_k_retrieval
                logger.info(f"BM25 retriever updated with {len(docs)} documents")
        
        except Exception as e:
            logger.error(f"Error updating BM25 retriever: {e}")
            self.bm25_retriever = None
    
    def _get_all_documents(self) -> List[Document]:
        """Get all documents from vectorstore."""
        try:
            if self.vectorstore is None:
                return []
            
            # FAISS doesn't directly expose documents, so we retrieve them via search
            # This is a workaround - in production, maintain a separate document store
            docstore = self.vectorstore.docstore
            docs = []
            
            if hasattr(docstore, '_dict'):
                docs = list(docstore._dict.values())
            
            return docs
        
        except Exception as e:
            logger.error(f"Error getting documents: {e}")
            return []
    
    def get_retriever(self, search_type: str = "hybrid"):
        """Get retriever with optional hybrid search and reranking."""
        if self.vectorstore is None:
            logger.error("Vectorstore not initialized")
            return None
        
        try:
            if search_type == "hybrid" and config.search.hybrid_search and self.bm25_retriever:
                # Hybrid search with semantic + BM25
                semantic_retriever = self.vectorstore.as_retriever(
                    search_kwargs={"k": config.search.top_k_retrieval}
                )
                
                ensemble_retriever = EnsembleRetriever(
                    retrievers=[semantic_retriever, self.bm25_retriever],
                    weights=[config.search.semantic_weight, 1 - config.search.semantic_weight]
                )
                
                # Add reranking if enabled
                if self.reranker:
                    logger.info("Using hybrid search with reranking")
                    return ContextualCompressionRetriever(
                        base_compressor=self.reranker,
                        base_retriever=ensemble_retriever
                    )
                
                logger.info("Using hybrid search")
                return ensemble_retriever
            
            else:
                # Standard semantic search
                retriever = self.vectorstore.as_retriever(
                    search_kwargs={"k": config.search.top_k_retrieval}
                )
                
                # Add reranking if enabled
                if self.reranker:
                    logger.info("Using semantic search with reranking")
                    return ContextualCompressionRetriever(
                        base_compressor=self.reranker,
                        base_retriever=retriever
                    )
                
                logger.info("Using semantic search")
                return retriever
        
        except Exception as e:
            logger.error(f"Error creating retriever: {e}")
            return None
    
    def search(self, query: str, k: int = None) -> List[Tuple[Document, float]]:
        """Search vectorstore and return documents with scores."""
        if self.vectorstore is None:
            logger.error("Vectorstore not initialized")
            return []
        
        k = k or config.search.top_k_rerank
        
        try:
            results = self.vectorstore.similarity_search_with_score(query, k=k)
            logger.info(f"Search returned {len(results)} results")
            return results
        
        except Exception as e:
            logger.error(f"Error during search: {e}")
            return []
    
    def save_vectorstore(self):
        """Save vectorstore to disk."""
        if self.vectorstore is None:
            logger.warning("No vectorstore to save")
            return
        
        try:
            persist_dir = Path(config.vectorstore.persist_directory)
            persist_dir.mkdir(parents=True, exist_ok=True)
            
            self.vectorstore.save_local(str(persist_dir))
            
            # Save BM25 retriever separately
            if self.bm25_retriever:
                bm25_path = persist_dir / "bm25_retriever.pkl"
                with open(bm25_path, 'wb') as f:
                    pickle.dump(self.bm25_retriever, f)
            
            logger.info(f"Vectorstore saved to {persist_dir}")
        
        except Exception as e:
            logger.error(f"Error saving vectorstore: {e}")
    
    def load_vectorstore(self) -> bool:
        """Load vectorstore from disk."""
        persist_dir = Path(config.vectorstore.persist_directory)
        
        if not persist_dir.exists() or not list(persist_dir.glob("*.faiss")):
            logger.info("No existing vectorstore found")
            return False
        
        try:
            logger.info(f"Loading vectorstore from {persist_dir}")
            self.vectorstore = FAISS.load_local(
                str(persist_dir),
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            
            # Load BM25 retriever if exists
            bm25_path = persist_dir / "bm25_retriever.pkl"
            if bm25_path.exists() and config.search.hybrid_search:
                try:
                    with open(bm25_path, 'rb') as f:
                        self.bm25_retriever = pickle.load(f)
                    logger.info("BM25 retriever loaded")
                except Exception as e:
                    logger.warning(f"Could not load BM25 retriever: {e}")
                    self._update_bm25_retriever()
            
            logger.info("Vectorstore loaded successfully")
            return True
        
        except Exception as e:
            logger.error(f"Error loading vectorstore: {e}")
            return False
    
    def clear_vectorstore(self):
        """Clear vectorstore and delete persisted data."""
        self.vectorstore = None
        self.bm25_retriever = None
        
        persist_dir = Path(config.vectorstore.persist_directory)
        if persist_dir.exists():
            try:
                import shutil
                shutil.rmtree(persist_dir)
                persist_dir.mkdir(parents=True, exist_ok=True)
                logger.info("Vectorstore cleared")
            except Exception as e:
                logger.error(f"Error clearing vectorstore: {e}")
    
    def get_statistics(self) -> dict:
        """Get vectorstore statistics."""
        if self.vectorstore is None:
            return {
                'total_vectors': 0,
                'dimension': 0,
                'index_type': 'None'
            }
        
        try:
            index = self.vectorstore.index
            return {
                'total_vectors': index.ntotal,
                'dimension': index.d,
                'index_type': type(index).__name__
            }
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {}
