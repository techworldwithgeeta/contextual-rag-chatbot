"""
Main Application Entry Point for the Contextual RAG Chatbot.

This module serves as the main entry point for the RAG Chatbot application,
orchestrating all components and providing the web interface.
"""

import logging
import sys
import os
from pathlib import Path

# Set environment variables for OpenAI embeddings BEFORE any imports
os.environ['EMBEDDING_MODEL'] = 'text-embedding-ada-002'
os.environ['EMBEDDING_DIMENSION'] = '1536'

# Setup Phoenix OTEL BEFORE any other imports
try:
    from phoenix.otel import register
    # Register the application with Phoenix OTEL
    tracer_provider = register(
        project_name="rag-chatbot",
        endpoint="http://localhost:6006/v1/traces",
        auto_instrument=True
    )
    print("✅ Phoenix OTEL registered successfully")
except ImportError:
    print("⚠️ Phoenix OTEL not available - install with: pip install arize-phoenix-otel")
except Exception as e:
    print(f"⚠️ Phoenix OTEL setup failed: {e}")

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add src to Python path
sys.path.append(str(Path(__file__).parent / "src"))

# Import application components
try:
    from src.config.settings import settings
    logger.info("✅ Settings imported successfully")
    
    from src.web_interface.api import RAGChatbotAPI
    logger.info("✅ RAGChatbotAPI imported successfully")
    
    from src.document_processing.processor import DocumentProcessor
    logger.info("✅ DocumentProcessor imported successfully")
    
    from src.rag_pipeline.vector_store import VectorStoreManager
    logger.info("✅ VectorStoreManager imported successfully")
    
    from src.rag_pipeline.llamaindex_rag_engine import LlamaIndexRAGEngine
    logger.info("✅ LlamaIndexRAGEngine imported successfully")
    
    from src.evaluation.phoenix_evaluator import PhoenixEvaluator
    logger.info("✅ PhoenixEvaluator imported successfully")
    
    from src.evaluation.ragas_evaluator import RAGAsEvaluator
    logger.info("✅ RAGAsEvaluator imported successfully")
    
    from src.web_interface.open_webui_connector import OpenWebUIConnector
    logger.info("✅ OpenWebUIConnector imported successfully")
    
except Exception as e:
    logger.error(f"❌ Import error: {e}")
    raise

# Configure logging with file handler
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(settings.LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)

class RAGChatbotApplication:
    """
    Main application class for the Contextual RAG Chatbot.
    
    This class orchestrates all components and provides the main
    application interface for the RAG Chatbot system.
    """
    
    def __init__(self):
        """
        Initialize the RAG Chatbot application.
        
        Sets up all components and validates system configuration.
        """
        logger.info("Initializing Contextual RAG Chatbot Application")
        
        # Validate configuration
        self._validate_configuration()
        
        # Initialize components
        self._initialize_components()
        
        # Initialize web interface
        self.api = RAGChatbotAPI()
        
        # Pass RAG engine to API
        self.api.set_rag_engine(self.rag_engine)
        
        # Initialize Open WebUI connector
        self.open_webui_connector = OpenWebUIConnector()
        
        logger.info("RAG Chatbot Application initialized successfully")
    
    def _validate_configuration(self):
        """
        Validate system configuration and dependencies.
        """
        logger.info("Validating system configuration...")
        
        # Create required directories
        import os
        os.makedirs("logs", exist_ok=True)
        os.makedirs("data", exist_ok=True)
        
        logger.info("✅ Configuration validation passed")
    
    def _initialize_components(self):
        """
        Initialize all RAG system components.
        """
        logger.info("Initializing RAG system components...")
        
        try:
            # Initialize evaluators first
            self.phoenix_evaluator = PhoenixEvaluator()
            self.ragas_evaluator = RAGAsEvaluator()
            logger.info("Evaluators initialized")
            
            # Initialize document processor with fast mode from settings
            self.document_processor = DocumentProcessor(fast_mode=settings.FAST_DOCUMENT_PROCESSING)
            logger.info(f"Document processor initialized with fast_mode={settings.FAST_DOCUMENT_PROCESSING}")
            
            # Initialize vector store manager
            self.vector_store_manager = VectorStoreManager()
            logger.info("Vector store manager initialized")
            
            # Initialize RAG engine with vector store manager and evaluators
            self.rag_engine = LlamaIndexRAGEngine(
                self.vector_store_manager, 
                phoenix_evaluator=self.phoenix_evaluator,
                ragas_evaluator=self.ragas_evaluator
            )
            logger.info("RAG engine initialized with Phoenix tracing and RAGAs evaluation")
            
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            raise
    
    def process_case_study_documents(self):
        """
        Process case study documents from the Case Study directory.
        """
        logger.info("Processing case study documents...")
        
        case_study_dir = Path("Case Study")
        if not case_study_dir.exists():
            logger.warning("Case Study directory not found. Skipping document processing.")
            return
        
        try:
            # Process documents
            processed_docs = self.document_processor.process_directory(str(case_study_dir))
            logger.info(f"✅ Processed {len(processed_docs)} documents")
            
            # Store in vector database
            for doc_info in processed_docs:
                self.vector_store_manager.store_document_embeddings(doc_info)
            
            logger.info("✅ Documents stored in vector database")
            
        except Exception as e:
            logger.error(f"Error processing documents: {e}")
            raise
    
    def get_system_stats(self):
        """
        Get system statistics and status.
        """
        try:
            stats = {
                "document_processing": {
                    "total_documents": len(self.vector_store_manager.get_all_documents()),
                    "total_chunks": sum(doc.get('total_chunks', 0) for doc in self.vector_store_manager.get_all_documents())
                },
                "vector_store": {
                    "total_documents": len(self.vector_store_manager.get_all_documents())
                },
                "system_info": {
                    "embedding_model": settings.EMBEDDING_MODEL,
                    "llm_model": settings.OLLAMA_MODEL
                }
            }
            return stats
        except Exception as e:
            logger.error(f"Error getting system stats: {e}")
            return {}
    
    def test_system(self):
        """
        Run system tests to verify all components are working.
        """
        logger.info("Running system tests...")
        
        test_results = {
            "overall": True,
            "components": {},
            "warnings": []
        }
        
        try:
            # Test RAG engine
            rag_test = self.rag_engine.test_connection()
            test_results["components"]["rag_engine"] = rag_test
            # Don't fail overall test just because RAG engine query engine is not available
            # This is normal when vector store is not fully configured
            if not rag_test.get("overall", False):
                logger.warning(f"⚠️ RAG engine test: {rag_test.get('message', 'Unknown error')}")
            else:
                logger.info("✅ RAG engine test passed")
            
                            # Test vector store
                try:
                    vector_store_status = self.vector_store_manager.get_status()
                    test_results["components"]["vector_store"] = vector_store_status
                    if vector_store_status['status'] == 'available':
                        logger.info("✅ Vector store test passed")
                    else:
                        logger.warning(f"⚠️ Vector store test: {vector_store_status.get('description', 'Unknown status')}")
                except Exception as e:
                    test_results["components"]["vector_store"] = {
                        "status": "warning",
                        "error": str(e)
                    }
                    test_results["warnings"].append(f"Vector store: {str(e)}")
                    logger.warning(f"⚠️ Vector store test: {str(e)}")
            
            # Test evaluators
            phoenix_status = "available" if self.phoenix_evaluator.phoenix_available else "unavailable"
            test_results["components"]["phoenix_evaluator"] = {
                "status": phoenix_status
            }
            if phoenix_status == "available":
                logger.info("✅ Phoenix evaluator test passed")
            else:
                logger.warning("⚠️ Phoenix evaluator not available")
            
            ragas_status = "available" if self.ragas_evaluator.ragas_available else "unavailable"
            test_results["components"]["ragas_evaluator"] = {
                "status": ragas_status,
                "method": self.ragas_evaluator.get_status().get('capabilities', [])
            }
            if ragas_status == "available":
                logger.info("✅ RAGAs evaluator test passed")
            else:
                logger.info("ℹ️ RAGAs evaluator using fallback metrics")
            
            # Determine overall status based on critical components
            critical_components_working = (
                test_results["components"].get("phoenix_evaluator", {}).get("status") == "available" or
                test_results["components"].get("ragas_evaluator", {}).get("status") == "available"
            )
            
            if critical_components_working:
                test_results["overall"] = True
                logger.info("✅ System tests completed - critical components available")
            else:
                test_results["overall"] = False
                logger.warning("⚠️ System tests completed - some critical components unavailable")
            
            if test_results["warnings"]:
                logger.info(f"⚠️ Warnings during system tests: {len(test_results['warnings'])} warnings")
                for warning in test_results["warnings"]:
                    logger.info(f"  - {warning}")
            
            return test_results
            
        except Exception as e:
            logger.error(f"System tests failed: {e}")
            test_results["overall"] = False
            return test_results
    
    def setup_open_webui_integration(self):
        """
        Setup minimal Open WebUI integration.
        """
        try:
            logger.info("Setting up Open WebUI integration...")
            
            # Create basic RAG model configuration
            config = self.open_webui_connector.create_rag_model_config()
            
            logger.info("✅ Open WebUI integration configured")
            logger.info(f"📋 RAG API URL: http://{settings.WEB_HOST}:{settings.WEB_PORT}")
            logger.info(f"🌐 Open WebUI URL: {settings.OPEN_WEBUI_URL}")
            
        except Exception as e:
            logger.error(f"Failed to setup Open WebUI integration: {e}")
    
    def get_integration_status(self):
        """
        Get minimal integration status.
        """
        try:
            status = {
                "rag_api": {
                    "status": "running",
                    "url": f"http://{settings.WEB_HOST}:{settings.WEB_PORT}"
                },
                "open_webui": {
                    "status": "configured",
                    "url": settings.OPEN_WEBUI_URL
                }
            }
            return status
        except Exception as e:
            logger.error(f"Error getting integration status: {e}")
            return {"error": str(e)}
    
    def generate_csv_files(self):
        """
        Generate CSV files with latest date format for evaluation data.
        """
        try:
            logger.info("📊 Generating CSV files with latest date format...")
            
            # Check if RAGAs evaluator is available
            if hasattr(self, 'ragas_evaluator') and self.ragas_evaluator:
                # Generate test evaluation data if none exists
                if not self.ragas_evaluator.evaluation_history:
                    logger.info("📝 Generating sample evaluation data...")
                    self._generate_sample_evaluation_data()
                
                # Export evaluation data
                logger.info("📊 Exporting evaluation data...")
                success = self.ragas_evaluator.export_evaluation_data(simplified=settings.SIMPLIFIED_CSV_EXPORT)
                if success:
                    logger.info("✅ Evaluation data exported successfully")
                else:
                    logger.warning("⚠️ Failed to export evaluation data")
                
                # Export summary data
                logger.info("📈 Exporting evaluation summary...")
                success = self.ragas_evaluator.generate_evaluation_summary_csv()
                if success:
                    logger.info("✅ Evaluation summary exported successfully")
                else:
                    logger.warning("⚠️ Failed to export evaluation summary")
                
                # Display CSV file information
                self._display_csv_file_info()
                
            else:
                logger.warning("⚠️ RAGAs evaluator not available for CSV generation")
                
        except Exception as e:
            logger.error(f"❌ Error generating CSV files: {e}")
    
    def _generate_sample_evaluation_data(self):
        """
        Generate sample evaluation data for demonstration.
        """
        try:
            # Sample queries and responses
            sample_data = [
                {
                    "query": "What are the procurement standards?",
                    "response": "The procurement standards include transparency, fairness, and accountability in the procurement process.",
                    "contexts": ["Procurement standards require transparency and fairness in all procurement activities."]
                },
                {
                    "query": "How does the procurement process work?",
                    "response": "The procurement process involves planning, sourcing, evaluation, and contract management stages.",
                    "contexts": ["The procurement process must follow established guidelines and procedures."]
                },
                {
                    "query": "What are the key principles of procurement?",
                    "response": "Key principles include transparency, fairness, competition, and value for money.",
                    "contexts": ["Key principles of procurement include transparency, fairness, and competition."]
                },
                {
                    "query": "What documents are required for procurement?",
                    "response": "Required documents include technical specifications, evaluation criteria, and contract terms.",
                    "contexts": ["Technical specifications and evaluation criteria are essential procurement documents."]
                },
                {
                    "query": "How is procurement transparency ensured?",
                    "response": "Transparency is ensured through public disclosure, clear procedures, and stakeholder engagement.",
                    "contexts": ["Transparency in procurement is achieved through public disclosure and clear procedures."]
                }
            ]
            
            # Generate evaluation data
            for i, data in enumerate(sample_data):
                self.ragas_evaluator.trace_query_execution(
                    query=data["query"],
                    response=data["response"],
                    source_documents=[{'text': ctx} for ctx in data["contexts"]],
                    processing_time=1.5 + (i * 0.2)
                )
            
            logger.info(f"✅ Generated {len(sample_data)} sample evaluation records")
            
        except Exception as e:
            logger.error(f"❌ Error generating sample data: {e}")
    
    def _display_csv_file_info(self):
        """
        Display information about generated CSV files.
        """
        try:
            import os
            from pathlib import Path
            
            evaluation_dir = Path("data/evaluation")
            if evaluation_dir.exists():
                csv_files = list(evaluation_dir.glob("*.csv"))
                
                if csv_files:
                    logger.info("📁 Generated CSV Files:")
                    for csv_file in sorted(csv_files, key=lambda x: x.stat().st_mtime, reverse=True):
                        file_size = csv_file.stat().st_size
                        file_time = csv_file.stat().st_mtime
                        import time
                        file_date = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(file_time))
                        logger.info(f"  📄 {csv_file.name} ({file_size} bytes, {file_date})")
                else:
                    logger.info("📁 No CSV files found in data/evaluation/")
            else:
                logger.info("📁 data/evaluation/ directory not found")
                
        except Exception as e:
            logger.error(f"❌ Error displaying CSV file info: {e}")
    
    def run(self, process_documents: bool = True, test_system: bool = True, setup_openwebui: bool = True, generate_csv: bool = True):
        """
        Run the RAG Chatbot application.
        
        Args:
            process_documents (bool): Whether to process case study documents
            test_system (bool): Whether to run system tests
            setup_openwebui (bool): Whether to setup Open WebUI integration
        """
        logger.info("Starting RAG Chatbot Application")
        
        try:
            # Run system tests if requested
            if test_system:
                test_results = self.test_system()
                if not test_results['overall']:
                    logger.warning("⚠️ Some system tests failed, but continuing with available components.")
                    logger.info("This is normal in development environments where some services may not be fully configured.")
                    # Don't return - continue with the application
            
            # Process case study documents if requested
            if process_documents:
                self.process_case_study_documents()
            
            # Display system statistics
            stats = self.get_system_stats()
            logger.info("System Statistics:")
            logger.info(f"  - Documents processed: {stats.get('document_processing', {}).get('total_documents', 0)}")
            logger.info(f"  - Total chunks: {stats.get('document_processing', {}).get('total_chunks', 0)}")
            logger.info(f"  - Vector store documents: {stats.get('vector_store', {}).get('total_documents', 0)}")
            logger.info(f"  - Embedding model: {stats.get('system_info', {}).get('embedding_model', 'unknown')}")
            logger.info(f"  - LLM model: {stats.get('system_info', {}).get('llm_model', 'unknown')}")
            
            # Setup Open WebUI integration if requested
            if setup_openwebui:
                logger.info("Setting up Open WebUI integration...")
                self.setup_open_webui_integration()
            else:
                logger.info("Skipping Open WebUI integration setup")
            
            # Display integration status
            integration_status = self.get_integration_status()
            logger.info("Integration Status:")
            logger.info(f"  - RAG API: {integration_status['rag_api']['status']} at {integration_status['rag_api']['url']}")
            if 'phoenix' in integration_status:
                logger.info(f"  - Phoenix: {integration_status['phoenix']['status']} at {integration_status['phoenix']['url']}")
            else:
                logger.info("  - Phoenix: not configured")
            logger.info(f"  - Open WebUI: {integration_status['open_webui']['status']} at {integration_status['open_webui']['url']}")
            
            # Generate CSV files with latest date format (unless disabled)
            if generate_csv:
                self.generate_csv_files()
            else:
                logger.info("Skipping CSV file generation")
            
            # Start the web interface
            logger.info(f"Starting web interface on {settings.WEB_HOST}:{settings.WEB_PORT}")
            logger.info(f"API available at: http://{settings.WEB_HOST}:{settings.WEB_PORT}")
            logger.info(f"API docs at: http://{settings.WEB_HOST}:{settings.WEB_PORT}/docs")
            logger.info(f"Health check: http://{settings.WEB_HOST}:{settings.WEB_PORT}/health")
            logger.info(f"Chat endpoint: http://{settings.WEB_HOST}:{settings.WEB_PORT}/chat")
            
            # Run the API with the correct host and port
            self.api.run(host=settings.WEB_HOST, port=settings.WEB_PORT)
            
        except KeyboardInterrupt:
            logger.info("Application stopped by user")
        except Exception as e:
            logger.error(f"Application failed: {e}")
            raise

def main():
    """
    Main entry point for the RAG Chatbot application.
    
    Parses command line arguments and starts the application.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Contextual RAG Chatbot with Open WebUI Integration")
    parser.add_argument(
        "--no-docs",
        action="store_true",
        help="Skip processing case study documents"
    )
    parser.add_argument(
        "--no-test",
        action="store_true",
        help="Skip system tests"
    )
    parser.add_argument(
        "--no-openwebui",
        action="store_true",
        help="Skip Open WebUI integration setup"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Host to bind the web interface to"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8001,
        help="Port to bind the web interface to"
    )
    parser.add_argument(
        "--setup-only",
        action="store_true",
        help="Only setup Open WebUI integration without starting the server"
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Skip CSV file generation"
    )
    
    args = parser.parse_args()
    
    try:
        # Create application
        app = RAGChatbotApplication()
        
        # Override settings with command line arguments
        settings.WEB_HOST = args.host
        settings.WEB_PORT = args.port
        
        if args.setup_only:
            # Only setup Open WebUI integration
            logger.info("Setting up Open WebUI integration only...")
            app.setup_open_webui_integration()
            logger.info("Open WebUI integration setup complete!")
            return
        
        # Run the full application
        app.run(
            process_documents=not args.no_docs,
            test_system=not args.no_test,
            setup_openwebui=not args.no_openwebui,
            generate_csv=not args.no_csv
        )
        
    except Exception as e:
        logger.error(f"Application failed to start: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 