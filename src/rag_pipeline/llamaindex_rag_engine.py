"""
LlamaIndex RAG engine module.
"""

import logging
import time
import uuid
from typing import Dict, Any, List

# Import LlamaIndex components
try:
    from llama_index.core import VectorStoreIndex, Settings
    from llama_index.llms.ollama import Ollama
    from llama_index.embeddings.openai import OpenAIEmbedding
    from llama_index.core.query_engine import RetrieverQueryEngine
    from llama_index.core.retrievers import VectorIndexRetriever
    PHOENIX_AVAILABLE = True
except ImportError:
    PHOENIX_AVAILABLE = False

# Import Crew.AI components
try:
    from crewai import Agent, Task, Crew, Process
    from crewai.tasks.task_output import TaskOutput
    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False

# Import Phoenix components
try:
    import phoenix as px
    PHOENIX_AVAILABLE = True
except ImportError:
    PHOENIX_AVAILABLE = False

from src.config.settings import settings

logger = logging.getLogger(__name__)

class LlamaIndexRAGEngine:
    """LlamaIndex RAG engine with Crew.AI integration."""
    
    def __init__(self, vector_store_manager=None, phoenix_evaluator=None, ragas_evaluator=None):
        """Initialize the RAG engine."""
        logger.info("LlamaIndex RAG engine initialized")
        
        self.vector_store_manager = vector_store_manager
        self.phoenix_evaluator = phoenix_evaluator
        self.ragas_evaluator = ragas_evaluator
        
        # Initialize components
        self._initialize_embedding_model()
        self._initialize_llm()
        self._initialize_vector_index()
        self._initialize_crew_manager()
        
        logger.info("✅ LlamaIndex RAG engine fully initialized")
    
    def _initialize_embedding_model(self):
        """Initialize the embedding model."""
        try:
            self.embedding_model = OpenAIEmbedding(
                model=settings.EMBEDDING_MODEL,
                api_key=settings.OPENAI_API_KEY
            )
            Settings.embed_model = self.embedding_model
            logger.info(f"✅ OpenAI embedding model initialized: {settings.EMBEDDING_MODEL}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize embedding model: {e}")
            raise
    
    def _initialize_llm(self):
        """Initialize the Ollama LLM."""
        try:
            # Create LlamaIndex LLM
            self.llm = Ollama(
                model=settings.OLLAMA_MODEL,
                request_timeout=60.0,
                temperature=0.7
            )
            Settings.llm = self.llm
            logger.info(f"✅ LlamaIndex LLM initialized: {settings.OLLAMA_MODEL}")
            
            # Create Crew.AI compatible LLM
            try:
                from langchain_ollama import OllamaLLM
                self.crew_llm = OllamaLLM(
                    model=settings.OLLAMA_MODEL,
                    base_url="http://localhost:11434"
                )
                logger.info(f"✅ Crew.AI LLM initialized: {settings.OLLAMA_MODEL}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to initialize Crew.AI LLM: {e}")
                self.crew_llm = None
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize LLM: {e}")
            raise
    
    def _initialize_vector_index(self):
        """Initialize the vector index from the vector store."""
        try:
            # Check if vector store manager and vector store are available
            if not self.vector_store_manager or not self.vector_store_manager.vector_store:
                logger.warning("⚠️ Vector store not available, creating fallback query engine")
                self.query_engine = None
                return
            
            # Get vector store
            vector_store = self.vector_store_manager.vector_store
            
            # Check if vector index is available
            if hasattr(self.vector_store_manager, 'vector_index') and self.vector_store_manager.vector_index:
                self.vector_index = self.vector_store_manager.vector_index
                self.query_engine = self.vector_index.as_query_engine(
                    similarity_top_k=settings.VECTOR_STORE_TOP_K
                )
                logger.info("✅ Vector index initialized from manager")
            else:
                # Create vector index from vector store
                self.vector_index = VectorStoreIndex.from_vector_store(vector_store)
                self.query_engine = self.vector_index.as_query_engine(
                    similarity_top_k=settings.VECTOR_STORE_TOP_K
                )
                logger.info("✅ Vector index initialized from vector store")
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize vector index: {e}")
            # Create a fallback query engine
            self.query_engine = None
    
    def _initialize_crew_manager(self):
        """Initialize Crew.AI manager."""
        if not CREWAI_AVAILABLE:
            logger.warning("⚠️ Crew.AI not available")
            self.crew_manager = None
            return
        
        try:
            # Import CrewAIManager
            from src.agents.crew_agents import CrewAIManager
            
            # Create Crew.AI manager without passing LLM - let it create its own
            self.crew_manager = CrewAIManager(llm=None)
            
            if self.crew_manager.agents:
                logger.info("✅ Crew.AI manager initialized with agents")
            else:
                logger.warning("⚠️ Crew.AI manager initialized but no agents available")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Crew.AI: {e}")
            self.crew_manager = None
    
    def query(self, query: str) -> Dict[str, Any]:
        """
        Process a query using RAG and Crew.AI.
        
        Args:
            query (str): User query
            
        Returns:
            Dict[str, Any]: Response with metadata
        """
        start_time = time.time()
        self._query_start_time = start_time  # Store for Phoenix tracing
        trace_id = str(uuid.uuid4())
        
        try:
            logger.info(f"Processing query: {query}")
            
            # Step 1: Basic RAG retrieval
            if self.query_engine:
                try:
                    response_text = self.query_engine.query(query)
                    source_documents = self._extract_source_documents(response_text)
                    logger.info(f"✅ Basic RAG response generated")
                except Exception as e:
                    logger.warning(f"⚠️ Basic RAG failed: {e}")
                    response_text = f"Unable to retrieve information: {str(e)}"
                    source_documents = []
            else:
                response_text = "Vector store not available"
                source_documents = []
            
            # Step 2: Crew.AI enhancement (optional)
            enhanced_response = response_text
            if settings.USE_CREW_AI_ENHANCEMENT and self.crew_manager and CREWAI_AVAILABLE:
                try:
                    enhanced_response = self._enhance_with_crew_ai(query, response_text, source_documents)
                    logger.info(f"✅ Crew.AI enhancement completed")
                except Exception as e:
                    logger.warning(f"⚠️ Crew.AI enhancement failed: {e}")
                    enhanced_response = response_text
            else:
                logger.info(f"✅ Using direct RAG response (Crew.AI enhancement disabled)")
            
            # Step 3: Phoenix tracing
            if PHOENIX_AVAILABLE and self.phoenix_evaluator:
                try:
                    self._trace_with_phoenix(query, str(enhanced_response), source_documents, trace_id)
                    logger.info(f"✅ Phoenix tracing completed")
                except Exception as e:
                    logger.warning(f"⚠️ Phoenix tracing failed: {e}")
            
            # Step 4: RAGAs evaluation
            if self.ragas_evaluator:
                try:
                    self._evaluate_with_ragas(query, str(enhanced_response), source_documents)
                    logger.info(f"✅ RAGAs evaluation completed")
                    
                    # Auto-generate CSV after evaluation (optional - can be disabled)
                    if settings.AUTO_GENERATE_CSV_AFTER_EVALUATION:
                        try:
                            self._auto_generate_csv_after_evaluation()
                            logger.info(f"✅ Auto CSV generation completed")
                        except Exception as csv_error:
                            logger.warning(f"⚠️ Auto CSV generation failed: {csv_error}")
                            
                except Exception as e:
                    logger.warning(f"⚠️ RAGAs evaluation failed: {e}")
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Prepare result data
            result_data = {
                'response': str(enhanced_response),
                'source_documents': source_documents,
                'processing_time': processing_time,
                'trace_id': trace_id,
                'model_used': settings.OLLAMA_MODEL,
                'query': query,
                'response_length': len(str(enhanced_response)),
                'source_count': len(source_documents),
                'framework': 'llamaindex',
                'crew_ai_enhanced': enhanced_response != response_text
            }
            
            logger.info(f"✅ Query processing completed in {processing_time:.2f}s")
            return result_data
            
        except Exception as e:
            logger.error(f"❌ Query processing failed: {e}")
            return {
                'response': f"Error processing query: {str(e)}",
                'source_documents': [],
                'processing_time': time.time() - start_time,
                'trace_id': trace_id,
                'model_used': settings.OLLAMA_MODEL,
                'query': query,
                'error': str(e),
                'framework': 'llamaindex'
            }
    
    def _extract_source_documents(self, response) -> List[Dict[str, Any]]:
        """Extract source documents from response."""
        try:
            if hasattr(response, 'source_nodes'):
                return [
                    {
                        'chunk_text': node.text,
                        'metadata': node.metadata,
                        'score': getattr(node, 'score', 0.0)
                    }
                    for node in response.source_nodes
                ]
            return []
        except Exception as e:
            logger.warning(f"Failed to extract source documents: {e}")
            return []
    
    def _enhance_with_crew_ai(self, query: str, initial_response: str, source_documents: List[Dict[str, Any]]) -> str:
        """Enhance response using Crew.AI agents."""
        try:
            # Use the Crew.AI manager to enhance the response
            enhanced_response = self.crew_manager.enhance_response(query, initial_response, source_documents)
            logger.info(f"✅ Crew.AI enhancement completed")
            return enhanced_response
            
        except Exception as e:
            logger.error(f"❌ Crew.AI enhancement failed: {e}")
            return initial_response
    
    def _trace_with_phoenix(self, query: str, response: str, source_documents: List[Dict[str, Any]], trace_id: str):
        """Trace the query with Phoenix."""
        try:
            if PHOENIX_AVAILABLE and self.phoenix_evaluator:
                # Calculate processing time for this trace
                processing_time = time.time() - getattr(self, '_query_start_time', time.time())
                
                # Use the Phoenix evaluator to trace the query
                evaluation_result = self.phoenix_evaluator.evaluate_response(
                    query=query,
                    response=response,
                    sources=source_documents,
                    trace_id=trace_id,
                    processing_time=processing_time
                )
                
                # Also log a separate query trace
                self.phoenix_evaluator.log_query_trace(
                    query=query,
                    response=response,
                    sources=source_documents,
                    trace_id=trace_id,
                    processing_time=processing_time,
                    model=settings.OLLAMA_MODEL
                )
                
                logger.info(f"✅ Phoenix tracing completed for trace_id: {trace_id}")
                return evaluation_result
                    
        except Exception as e:
            logger.warning(f"Phoenix tracing failed: {e}")
            return None
    
    def _evaluate_with_ragas(self, query: str, response: str, source_documents: List[Dict[str, Any]]):
        """Evaluate the response with RAGAs."""
        try:
            if self.ragas_evaluator:
                # Use the RAGAs evaluator to evaluate the response
                evaluation_result = self.ragas_evaluator.evaluate_response(
                    query=query,
                    response=response,
                    sources=source_documents
                )
                logger.info(f"✅ RAGAs evaluation completed: {evaluation_result.get('method', 'unknown')}")
                    
        except Exception as e:
            logger.warning(f"RAGAs evaluation failed: {e}")
    
    def _auto_generate_csv_after_evaluation(self):
        """Auto-generate CSV files after evaluation."""
        try:
            if self.ragas_evaluator:
                # Export evaluation data with simplified fields by default
                self.ragas_evaluator.export_evaluation_data(simplified=settings.SIMPLIFIED_CSV_EXPORT)
                
                # Export summary data
                self.ragas_evaluator.generate_evaluation_summary_csv()
                
                logger.info(f"✅ Auto CSV generation completed after evaluation")
                
        except Exception as e:
            logger.warning(f"Auto CSV generation failed: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get engine status and capabilities."""
        return {
            'framework': 'llamaindex',
            'phoenix_available': PHOENIX_AVAILABLE and self.phoenix_evaluator is not None,
            'ragas_available': self.ragas_evaluator is not None and self.ragas_evaluator.ragas_available,
            'crew_ai_available': CREWAI_AVAILABLE and self.crew_manager is not None,
            'llm_model': settings.OLLAMA_MODEL,
            'embedding_model': settings.EMBEDDING_MODEL,
            'vector_store': 'pgvector',
            'database_url': settings.DATABASE_URL,
            'capabilities': [
                'llamaindex_rag',
                'ollama_llm',
                'crew_ai_agents',
                'phoenix_tracing',
                'ragas_evaluation',
                'open_webui_integration',
                'pgvector_storage'
            ]
        }
    
    def test_connection(self) -> Dict[str, Any]:
        """
        Test the connection of the LlamaIndex RAG engine.
        
        Returns:
            Dict[str, Any]: Connection status.
        """
        try:
            # Check if query engine is available
            if self.query_engine is None:
                return {
                    "overall": False, 
                    "message": "RAG engine query engine not available - vector store may not be initialized"
                }
            
            # Attempt a simple query to test the engine
            self.query_engine.query("test query")
            return {"overall": True, "message": "RAG engine connected successfully"}
        except Exception as e:
            return {"overall": False, "message": f"RAG engine connection failed: {e}"} 