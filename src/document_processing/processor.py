"""
Document processing module using Docling for PDF to Markdown conversion.
"""

import logging
import os
from typing import List, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Document processor using Docling for PDF to Markdown conversion."""
    
    def __init__(self, fast_mode: bool = True):
        """Initialize the document processor.
        
        Args:
            fast_mode: If True, uses faster processing options and disables heavy OCR engines
        """
        self.fast_mode = fast_mode
        logger.info(f"Document processor initialized with Docling (fast_mode={fast_mode})")
        
        # Check if Docling is available
        try:
            import docling
            self.docling_available = True
            logger.info("✅ Docling available for document processing")
            
            # Configure for faster processing if fast_mode is enabled
            if self.fast_mode:
                self._configure_fast_mode()
                
        except ImportError:
            self.docling_available = False
            logger.error("❌ Docling not available. Install with: pip install docling")
    
    def _configure_fast_mode(self):
        """Configure docling for faster processing."""
        try:
            import os
            # Disable heavy OCR engines for faster startup
            os.environ['DOCLING_DISABLE_OCR'] = '1'
            # Use CPU only to avoid GPU initialization delays
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            logger.info("🔧 Configured fast mode: disabled OCR engines and GPU")
        except Exception as e:
            logger.warning(f"⚠️ Could not configure fast mode: {e}")
    
    def process_documents(self, documents: List[str]) -> List[Dict[str, Any]]:
        """Process documents using Docling and return chunks."""
        logger.info(f"Processing {len(documents)} documents with Docling")
        processed_chunks = []
        
        for doc_path in documents:
            try:
                chunks = self._process_single_document(doc_path)
                processed_chunks.extend(chunks)
                logger.info(f"✅ Processed {doc_path}: {len(chunks)} chunks")
            except Exception as e:
                logger.error(f"❌ Failed to process {doc_path}: {e}")
        
        return processed_chunks
    
    def process_directory(self, directory_path: str) -> List[Dict[str, Any]]:
        """Process all documents in a directory using Docling."""
        logger.info(f"Processing directory with Docling: {directory_path}")
        
        if not os.path.exists(directory_path):
            logger.error(f"❌ Directory not found: {directory_path}")
            return []
        
        # Find all supported files
        supported_extensions = {'.pdf', '.docx', '.txt', '.md'}
        documents = []
        
        for file_path in Path(directory_path).rglob('*'):
            if file_path.suffix.lower() in supported_extensions:
                documents.append(str(file_path))
        
        logger.info(f"Found {len(documents)} documents to process with Docling")
        return self.process_documents(documents)
    
    def _process_single_document(self, file_path: str) -> List[Dict[str, Any]]:
        """Process a single document using Docling."""
        file_path = Path(file_path)
        extension = file_path.suffix.lower()
        
        try:
            if extension == '.pdf':
                return self._process_pdf_with_docling(file_path)
            elif extension == '.docx':
                return self._process_docx_with_docling(file_path)
            elif extension in ['.txt', '.md']:
                return self._process_text(file_path)
            else:
                logger.warning(f"⚠️ Unsupported file type: {extension}")
                return []
        except Exception as e:
            logger.error(f"❌ Error processing {file_path}: {e}")
            return []
    
    def _process_pdf_with_docling(self, file_path: Path) -> List[Dict[str, Any]]:
        """Process PDF files using Docling for PDF to Markdown conversion."""
        if not self.docling_available:
            logger.error("❌ Docling not available for PDF processing")
            return []
        
        try:
            from docling.document_converter import DocumentConverter
            from docling.datamodel.base_models import InputFormat
            
            logger.info(f"🔄 Converting PDF to Markdown with Docling: {file_path.name}")
            
            # Create DocumentConverter with optimized settings
            converter_kwargs = {
                'raises_on_error': False,
            }
            
            # Add fast mode options
            if self.fast_mode:
                converter_kwargs.update({
                    'ocr_enabled': False,  # Disable OCR for speed
                    'extract_images': False,  # Disable image extraction
                })
            
            converter = DocumentConverter(**converter_kwargs)
            
            # Convert PDF to markdown using Docling
            result = converter.convert(
                source=str(file_path),
                raises_on_error=False
            )
            
            if result and hasattr(result, 'content') and result.content:
                markdown_content = result.content
                # Split markdown into chunks
                chunks = self._split_markdown_into_chunks(markdown_content, file_path)
                logger.info(f"✅ PDF converted to Markdown: {file_path.name} - {len(chunks)} chunks")
                return chunks
            else:
                logger.warning(f"⚠️ No content extracted from PDF: {file_path.name}")
                return []
                
        except Exception as e:
            logger.error(f"❌ Docling PDF processing error: {e}")
            return []
    
    def _process_docx_with_docling(self, file_path: Path) -> List[Dict[str, Any]]:
        """Process DOCX files using Docling."""
        if not self.docling_available:
            logger.error("❌ Docling not available for DOCX processing")
            return []
        
        try:
            from docling.document_converter import DocumentConverter
            
            logger.info(f"🔄 Converting DOCX with Docling: {file_path.name}")
            
            # Create DocumentConverter with optimized settings
            converter_kwargs = {
                'raises_on_error': False,
            }
            
            # Add fast mode options
            if self.fast_mode:
                converter_kwargs.update({
                    'ocr_enabled': False,  # Disable OCR for speed
                    'extract_images': False,  # Disable image extraction
                })
            
            converter = DocumentConverter(**converter_kwargs)
            
            # Convert DOCX to markdown using Docling
            result = converter.convert(
                source=str(file_path),
                raises_on_error=False
            )
            
            if result and hasattr(result, 'content') and result.content:
                markdown_content = result.content
                # Split markdown into chunks
                chunks = self._split_markdown_into_chunks(markdown_content, file_path)
                logger.info(f"✅ DOCX converted to Markdown: {file_path.name} - {len(chunks)} chunks")
                return chunks
            else:
                logger.warning(f"⚠️ No content extracted from DOCX: {file_path.name}")
                return []
                
        except Exception as e:
            logger.error(f"❌ Docling DOCX processing error: {e}")
            return []
    
    def _process_text(self, file_path: Path) -> List[Dict[str, Any]]:
        """Process text files."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
            
            if text.strip():
                chunk = {
                    'text': text,
                    'metadata': {
                        'source': str(file_path),
                        'file_type': file_path.suffix.lower(),
                        'size': len(text),
                        'processed_by': 'docling_text'
                    }
                }
                logger.info(f"✅ Text file processed: {file_path.name}")
                return [chunk]
            else:
                logger.warning(f"⚠️ Empty text file: {file_path.name}")
                return []
                
        except Exception as e:
            logger.error(f"❌ Text file processing error: {e}")
            return []
    
    def _split_markdown_into_chunks(self, markdown_content: str, file_path: Path) -> List[Dict[str, Any]]:
        """Split markdown content into manageable chunks for vector storage."""
        try:
            # Split by headers (##, ###, etc.)
            import re
            
            # Split by markdown headers
            header_pattern = r'^#{1,6}\s+.+$'
            sections = re.split(header_pattern, markdown_content, flags=re.MULTILINE)
            headers = re.findall(header_pattern, markdown_content, flags=re.MULTILINE)
            
            chunks = []
            
            # Process each section
            for i, section in enumerate(sections):
                section = section.strip()
                if not section:
                    continue
                
                # Get the header for this section
                header = headers[i-1] if i > 0 and i-1 < len(headers) else "Introduction"
                
                # Split section into smaller chunks if too long
                section_chunks = self._split_long_text(section, max_length=1000)
                
                for j, chunk_text in enumerate(section_chunks):
                    chunk = {
                        'text': chunk_text,
                        'metadata': {
                            'source': str(file_path),
                            'file_type': 'pdf_to_markdown',
                            'section_header': header,
                            'chunk_index': j,
                            'total_chunks': len(section_chunks),
                            'processed_by': 'docling',
                            'original_file': file_path.name
                        }
                    }
                    chunks.append(chunk)
            
            # If no headers found, split by paragraphs
            if not chunks:
                paragraphs = markdown_content.split('\n\n')
                for i, paragraph in enumerate(paragraphs):
                    paragraph = paragraph.strip()
                    if paragraph:
                        chunk = {
                            'text': paragraph,
                            'metadata': {
                                'source': str(file_path),
                                'file_type': 'pdf_to_markdown',
                                'section_header': 'Paragraph',
                                'chunk_index': i,
                                'processed_by': 'docling',
                                'original_file': file_path.name
                            }
                        }
                        chunks.append(chunk)
            
            return chunks
            
        except Exception as e:
            logger.error(f"❌ Error splitting markdown: {e}")
            # Fallback: return the entire content as one chunk
            return [{
                'text': markdown_content,
                'metadata': {
                    'source': str(file_path),
                    'file_type': 'pdf_to_markdown',
                    'section_header': 'Full Document',
                    'processed_by': 'docling',
                    'original_file': file_path.name
                }
            }]
    
    def _split_long_text(self, text: str, max_length: int = 1000) -> List[str]:
        """Split long text into smaller chunks."""
        if len(text) <= max_length:
            return [text]
        
        chunks = []
        current_chunk = ""
        
        for sentence in text.split('.'):
            sentence = sentence.strip() + '.'
            if len(current_chunk + sentence) <= max_length:
                current_chunk += sentence + ' '
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ' '
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks 