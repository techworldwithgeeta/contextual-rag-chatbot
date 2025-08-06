#!/usr/bin/env python3
"""
Document Processing Script for the Contextual RAG Chatbot.

This script processes the case study documents and prepares them
for the RAG system by extracting text, creating embeddings, and
storing them in the vector database.
"""

import logging
import sys
from pathlib import Path

# Add src to Python path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.config.settings import settings
from src.document_processing.processor import DocumentProcessor
from src.rag_pipeline.vector_store import VectorStoreManager

# Configure logging
logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL))
logger = logging.getLogger(__name__)

def process_case_study_documents():
    """
    Process all documents in the Case Study directory.
    
    Extracts text from documents, creates chunks, generates embeddings,
    and stores them in the vector database.
    """
    logger.info("Starting document processing...")
    
    # Initialize components with fast mode from settings
    document_processor = DocumentProcessor(fast_mode=settings.FAST_DOCUMENT_PROCESSING)
    vector_store_manager = VectorStoreManager()
    
    # Check if Case Study directory exists
    case_study_dir = Path("Case Study")
    if not case_study_dir.exists():
        logger.error("Case Study directory not found!")
        logger.error("Please ensure the Case Study directory exists with your documents.")
        return False
    
    # Get list of documents
    documents = list(case_study_dir.glob("*"))
    supported_extensions = {'.pdf', '.docx', '.txt'}
    
    # Filter supported documents
    supported_docs = [doc for doc in documents if doc.is_file() and doc.suffix.lower() in supported_extensions]
    
    if not supported_docs:
        logger.error("No supported documents found in Case Study directory!")
        logger.info(f"Supported formats: {', '.join(supported_extensions)}")
        return False
    
    logger.info(f"Found {len(supported_docs)} supported documents:")
    for doc in supported_docs:
        logger.info(f"  - {doc.name} ({doc.suffix})")
    
    # Process each document
    processed_count = 0
    skipped_count = 0
    failed_count = 0
    
    for doc_path in supported_docs:
        try:
            logger.info(f"Processing document: {doc_path.name}")
            
            # Process document
            processed_document = document_processor.process_document(str(doc_path))
            
            # Store embeddings
            success = vector_store_manager.store_document_embeddings(processed_document)
            
            if success:
                # Check if document was skipped (already exists)
                if vector_store_manager.document_exists(processed_document['document_hash']):
                    logger.info(f"Skipped (already exists): {doc_path.name}")
                    skipped_count += 1
                else:
                    logger.info(f"Successfully processed: {doc_path.name} ({processed_document['total_chunks']} chunks)")
                    processed_count += 1
            else:
                logger.error(f"Failed to store embeddings for: {doc_path.name}")
                failed_count += 1
                
        except Exception as e:
            logger.error(f"Error processing {doc_path.name}: {e}")
            failed_count += 1
    
    # Summary
    logger.info("Document processing completed!")
    logger.info(f"  - Successfully processed: {processed_count} documents")
    logger.info(f"  - Skipped (already exists): {skipped_count} documents")
    logger.info(f"  - Failed: {failed_count} documents")
    
    return processed_count > 0

def get_processing_stats():
    """
    Get statistics about processed documents.
    
    Returns:
        Dict: Processing statistics
    """
    try:
        document_processor = DocumentProcessor()
        vector_store_manager = VectorStoreManager()
        
        doc_stats = document_processor.get_processing_stats()
        vector_stats = vector_store_manager.get_vector_store_stats()
        
        stats = {
            'document_processing': doc_stats,
            'vector_store': vector_stats,
            'summary': {
                'total_documents': doc_stats.get('total_documents', 0),
                'total_chunks': doc_stats.get('total_chunks', 0),
                'cache_size_mb': doc_stats.get('cache_size_mb', 0),
                'vector_documents': vector_stats.get('total_documents', 0),
                'vector_embeddings': vector_stats.get('total_embeddings', 0)
            }
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting processing stats: {e}")
        return {}

def list_processed_documents():
    """
    List all processed documents with their details.
    """
    try:
        vector_store_manager = VectorStoreManager()
        documents = vector_store_manager.get_all_documents()
        
        if not documents:
            logger.info("No documents have been processed yet.")
            return
        
        logger.info(f"Processed Documents ({len(documents)} total):")
        logger.info("-" * 80)
        
        for i, doc in enumerate(documents, 1):
            logger.info(f"{i}. {doc['file_name']}")
            logger.info(f"   - Type: {doc['file_type']}")
            logger.info(f"   - Size: {doc['file_size']:,} bytes")
            logger.info(f"   - Chunks: {doc['total_chunks']}")
            logger.info(f"   - Processed: {doc['processing_timestamp']}")
            logger.info(f"   - Hash: {doc['document_hash'][:16]}...")
            logger.info("")
        
    except Exception as e:
        logger.error(f"Error listing documents: {e}")

def clear_processed_documents():
    """
    Clear all processed documents and start fresh.
    """
    try:
        logger.warning("This will clear all processed documents and embeddings!")
        response = input("Are you sure you want to continue? (yes/no): ")
        
        if response.lower() != 'yes':
            logger.info("Operation cancelled.")
            return
        
        vector_store_manager = VectorStoreManager()
        success = vector_store_manager.clear_all_data()
        
        if success:
            logger.info("Successfully cleared all processed documents.")
        else:
            logger.error("Failed to clear processed documents.")
            
    except Exception as e:
        logger.error(f"Error clearing documents: {e}")

def main():
    """
    Main function for document processing.
    
    Provides a command-line interface for document processing operations.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Document Processing for RAG Chatbot")
    parser.add_argument(
        "--action",
        choices=['process', 'stats', 'list', 'clear'],
        default='process',
        help="Action to perform"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        if args.action == 'process':
            success = process_case_study_documents()
            if success:
                logger.info("Document processing completed successfully!")
                # Show stats after processing
                stats = get_processing_stats()
                logger.info("Processing Statistics:")
                logger.info(f"  - Documents: {stats.get('summary', {}).get('total_documents', 0)}")
                logger.info(f"  - Chunks: {stats.get('summary', {}).get('total_chunks', 0)}")
                logger.info(f"  - Cache Size: {stats.get('summary', {}).get('cache_size_mb', 0)} MB")
            else:
                logger.error("Document processing failed!")
                sys.exit(1)
                
        elif args.action == 'stats':
            stats = get_processing_stats()
            logger.info("Document Processing Statistics:")
            logger.info(f"  - Total Documents: {stats.get('summary', {}).get('total_documents', 0)}")
            logger.info(f"  - Total Chunks: {stats.get('summary', {}).get('total_chunks', 0)}")
            logger.info(f"  - Cache Size: {stats.get('summary', {}).get('cache_size_mb', 0)} MB")
            logger.info(f"  - Vector Documents: {stats.get('summary', {}).get('vector_documents', 0)}")
            logger.info(f"  - Vector Embeddings: {stats.get('summary', {}).get('vector_embeddings', 0)}")
            
        elif args.action == 'list':
            list_processed_documents()
            
        elif args.action == 'clear':
            clear_processed_documents()
        
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user.")
    except Exception as e:
        logger.error(f"Document processing failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 