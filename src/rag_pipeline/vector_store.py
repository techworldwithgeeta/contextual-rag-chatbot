"""
Vector store management module.
"""

import logging
from typing import Dict, Any, List
import os

# Set up logger first
logger = logging.getLogger(__name__)

# Import LlamaIndex components
try:
    from llama_index.core import VectorStoreIndex, Document
    from llama_index.core.storage.storage_context import StorageContext
    from llama_index.embeddings.openai import OpenAIEmbedding
    from llama_index.vector_stores.postgres import PGVectorStore
    VECTOR_STORE_AVAILABLE = True
    LLAMAINDEX_IMPORTS_SUCCESSFUL = True
except ImportError as e:
    logger.warning(f"⚠️ LlamaIndex imports failed: {e}")
    VECTOR_STORE_AVAILABLE = False
    LLAMAINDEX_IMPORTS_SUCCESSFUL = False

from src.config.settings import settings

class VectorStoreManager:
    """Vector store manager for document storage and retrieval."""
    
    def __init__(self):
        """Initialize the vector store manager."""
        logger.info("Vector store manager initialized")
        
        # Always try to initialize vector store
        self._initialize_vector_store()
        
        if not self.vector_store:
            logger.warning("⚠️ Vector store components not available")
    
    def _initialize_vector_store(self):
        """Initialize the vector store."""
        # Check if LlamaIndex imports are available
        if not LLAMAINDEX_IMPORTS_SUCCESSFUL:
            logger.warning("⚠️ LlamaIndex components not available, skipping vector store initialization")
            self.vector_store = None
            self.vector_index = None
            return
        
        try:
            # Initialize OpenAI embedding model
            embed_model = OpenAIEmbedding(
                model=settings.EMBEDDING_MODEL,
                api_key=settings.OPENAI_API_KEY
            )
            logger.info(f"✅ OpenAI embedding model initialized: {settings.EMBEDDING_MODEL}")
            
            # Initialize PGVector store
            try:
                # Parse database URL to extract connection details
                from urllib.parse import urlparse
                db_url = urlparse(settings.DATABASE_URL)
                
                # Create PGVector store
                self.vector_store = PGVectorStore.from_params(
                    database=db_url.path[1:],  # Remove leading slash
                    host=db_url.hostname,
                    port=db_url.port or 5432,
                    user=db_url.username,
                    password=db_url.password,
                    table_name="document_vectors",
                    embed_dim=settings.EMBEDDING_DIMENSION
                )
                logger.info("✅ PGVector store initialized successfully")
                
            except Exception as pg_error:
                logger.warning(f"⚠️ PGVector initialization failed: {pg_error}")
                logger.info("🔄 Falling back to in-memory vector store")
                
                # Fallback to in-memory store
                storage_context = StorageContext.from_defaults()
                empty_doc = Document(text="Initial document for vector store initialization")
                self.vector_index = VectorStoreIndex.from_documents(
                    [empty_doc],
                    storage_context=storage_context
                )
                self.vector_store = self.vector_index.vector_store
                logger.info("✅ In-memory vector store initialized as fallback")
            
            # Create storage context with PGVector store
            storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store
            )
            logger.info("✅ Storage context created with PGVector")
            
            # Create vector index from PGVector store
            self.vector_index = VectorStoreIndex.from_vector_store(
                self.vector_store,
                storage_context=storage_context
            )
            logger.info("✅ Vector index created from PGVector store")
            
            logger.info("✅ PGVector-based vector store initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize vector store: {e}")
            self.vector_store = None
            self.vector_index = None
    
    def store_document_embeddings(self, doc_info: Dict[str, Any]):
        """Store document embeddings."""
        logger.info(f"Storing document embeddings: {doc_info.get('filename', 'unknown')}")
        
        if self.vector_store and self.vector_index and LLAMAINDEX_IMPORTS_SUCCESSFUL:
            try:
                # Create document
                document = Document(
                    text=doc_info.get('content', ''),
                    metadata=doc_info.get('metadata', {})
                )
                
                # Insert into vector store
                self.vector_index.insert(document)
                logger.info(f"✅ Document stored in vector store")
                
            except Exception as e:
                logger.error(f"❌ Failed to store document: {e}")
        else:
            logger.warning("⚠️ Vector store not available, skipping document storage")
    
    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Get all documents."""
        if self.vector_store and self.vector_index and LLAMAINDEX_IMPORTS_SUCCESSFUL:
            try:
                # This is a simplified version - in practice you'd query the vector store
                return []
            except Exception as e:
                logger.error(f"❌ Failed to get documents: {e}")
                return []
        else:
            logger.warning("⚠️ Vector store not available, returning empty document list")
            return []
    
    def get_status(self) -> Dict[str, Any]:
        """Get vector store status."""
        return {
            'status': 'available' if self.vector_store is not None else 'unavailable',
            'type': 'pgvector' if hasattr(self, 'vector_store') and self.vector_store is not None else None,
            'description': 'PGVector PostgreSQL vector store for document storage',
            'database_url': settings.DATABASE_URL,
            'embedding_model': settings.EMBEDDING_MODEL,
            'embedding_dimension': settings.EMBEDDING_DIMENSION
        } 