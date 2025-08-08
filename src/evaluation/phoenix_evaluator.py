"""
Phoenix evaluation module using proper Phoenix OTEL (OpenTelemetry) setup.
"""

import logging
import time
import uuid
import os
import json
import csv
from typing import Dict, Any, List, Optional
from datetime import datetime

# Try to import Phoenix OTEL components
try:
    from phoenix.otel import register
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode
    PHOENIX_OTEL_AVAILABLE = True
except ImportError:
    PHOENIX_OTEL_AVAILABLE = False
    register = None
    trace = None
    Status = None
    StatusCode = None

logger = logging.getLogger(__name__)

class PhoenixEvaluator:
    """Phoenix evaluator using proper Phoenix OTEL for RAG tracing and evaluation."""
    
    def __init__(self):
        """Initialize the Phoenix evaluator with OTEL setup."""
        logger.info("Phoenix evaluator initialized with OTEL")
        self.phoenix_available = PHOENIX_OTEL_AVAILABLE
        self.tracer = None
        self.project_name = "rag-chatbot"
        
        # Setup logging directories
        self.log_dir = "data/phoenix_logs"
        self.csv_dir = "data/phoenix_csv"
        self._setup_directories()
        
        if self.phoenix_available:
            try:
                # Setup Phoenix OTEL with proper configuration
                self._setup_phoenix_otel()
                logger.info("✅ Phoenix OTEL setup completed successfully")
                
            except Exception as e:
                logger.warning(f"⚠️ Phoenix OTEL setup failed: {e}")
                self.phoenix_available = False
        else:
            logger.warning("⚠️ Phoenix OTEL not available - install with: pip install arize-phoenix-otel")
    
    def _setup_phoenix_otel(self):
        """Setup Phoenix OTEL with proper configuration."""
        try:
            # Get Phoenix collector endpoint from environment or use default
            phoenix_endpoint = os.getenv('PHOENIX_COLLECTOR_ENDPOINT', 'http://localhost:6006')
            
            # Register the application with Phoenix OTEL
            # This should be called BEFORE any code execution
            tracer_provider = register(
                project_name=self.project_name,
                endpoint=f"{phoenix_endpoint}/v1/traces",
                auto_instrument=True
            )
            
            # Get the tracer for this application
            self.tracer = trace.get_tracer(__name__)
            
            logger.info(f"✅ Phoenix OTEL registered with endpoint: {phoenix_endpoint}")
            logger.info(f"✅ Tracer initialized: {self.tracer}")
            
        except Exception as e:
            logger.error(f"❌ Phoenix OTEL setup failed: {e}")
            raise
    
    def _setup_directories(self):
        """Setup directories for Phoenix logs and CSV files."""
        try:
            os.makedirs(self.log_dir, exist_ok=True)
            os.makedirs(self.csv_dir, exist_ok=True)
            logger.info(f"✅ Phoenix directories created: {self.log_dir}, {self.csv_dir}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to create Phoenix directories: {e}")
    
    def evaluate_response(self, query: str, response: str, sources: List[Dict[str, Any]], 
                         trace_id: Optional[str] = None, processing_time: float = 0.0) -> Dict[str, Any]:
        """Evaluate a response using Phoenix OTEL and send traces."""
        if not self.phoenix_available or not self.tracer:
            logger.warning("⚠️ Phoenix OTEL not available for evaluation")
            return self._fallback_evaluation(query, response, sources)
        
        try:
            # Generate trace ID if not provided
            if not trace_id:
                trace_id = str(uuid.uuid4())
            
            # Create evaluation metrics
            evaluation_metrics = self._calculate_metrics(query, response, sources)
            
            # Create a span for this evaluation
            with self.tracer.start_as_current_span(
                "rag_response_evaluation",
                attributes={
                    "trace_id": trace_id,
                    "query": query,
                    "response_length": len(response),
                    "sources_count": len(sources),
                    "processing_time": processing_time,
                    "model": "llama2:7b",
                    "framework": "llamaindex",
                    "evaluation_type": "phoenix_otel"
                }
            ) as span:
                # Add evaluation metrics as span attributes
                span.set_attributes({
                    "overall_score": evaluation_metrics.get("overall_score", 0.0),
                    "relevance_score": evaluation_metrics.get("relevance_score", 0.0),
                    "completeness_score": evaluation_metrics.get("completeness_score", 0.0),
                    "source_utilization": evaluation_metrics.get("source_utilization", 0.0),
                    "query_length": evaluation_metrics.get("query_length", 0),
                    "response_length_metrics": evaluation_metrics.get("response_length", 0),
                    "sources_count_metrics": evaluation_metrics.get("sources_count", 0)
                })
                
                # Set span status to OK
                span.set_status(Status(StatusCode.OK))
                
                # Log evaluation completion
                logger.info(f"✅ Phoenix OTEL evaluation completed for trace_id: {trace_id}")
                logger.info(f"📊 Phoenix metrics: {evaluation_metrics}")
                
                # Save trace data to files for backup
                self._save_trace_data(query, response, sources, trace_id, processing_time, evaluation_metrics)
                
                return {
                    "trace_id": trace_id,
                    "score": evaluation_metrics.get("overall_score", 0.0),
                    "metrics": evaluation_metrics,
                    "phoenix_logged": True,
                    "otel_span_created": True
                }
            
        except Exception as e:
            logger.error(f"❌ Phoenix OTEL evaluation failed: {e}")
            return self._fallback_evaluation(query, response, sources)
    
    def log_query_trace(self, query: str, response: str, sources: List[Dict[str, Any]], 
                       trace_id: str, processing_time: float, model: str = "llama2:7b"):
        """Log a query trace using Phoenix OTEL."""
        if not self.phoenix_available or not self.tracer:
            logger.warning("⚠️ Phoenix OTEL not available for tracing")
            return False
        
        try:
            # Create a span for query processing
            with self.tracer.start_as_current_span(
                "rag_query_processing",
                attributes={
                    "trace_id": trace_id,
                    "query": query,
                    "response_length": len(response),
                    "sources_count": len(sources),
                    "processing_time": processing_time,
                    "model": model,
                    "framework": "llamaindex"
                }
            ) as span:
                # Add additional attributes
                span.set_attributes({
                    "query_type": "rag_query",
                    "response_generated": True,
                    "sources_retrieved": len(sources) > 0
                })
                
                # Set span status to OK
                span.set_status(Status(StatusCode.OK))
                
                logger.info(f"✅ Query trace logged to Phoenix OTEL: {trace_id}")
                logger.info(f"📝 Query: {query[:100]}...")
                logger.info(f"📊 Processing time: {processing_time:.2f}s")
                logger.info(f"📄 Sources count: {len(sources)}")
                
                return True
                
        except Exception as e:
            logger.error(f"❌ Query trace logging failed: {e}")
            return False
    
    def _calculate_metrics(self, query: str, response: str, sources: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        try:
            # Basic metrics calculation
            query_length = len(query.split())
            response_length = len(response.split())
            sources_count = len(sources)
            
            # Relevance score (simple word overlap)
            query_words = set(query.lower().split())
            response_words = set(response.lower().split())
            
            if query_words:
                relevance_score = len(query_words.intersection(response_words)) / len(query_words)
            else:
                relevance_score = 0.0
            
            # Completeness score (based on response length)
            completeness_score = min(response_length / 50, 1.0)  # Normalize to 0-1
            
            # Source utilization score
            source_utilization = min(sources_count / 3, 1.0)  # Normalize to 0-1
            
            # Overall score (weighted average)
            overall_score = (
                relevance_score * 0.4 +
                completeness_score * 0.3 +
                source_utilization * 0.3
            )
            
            return {
                "overall_score": round(overall_score, 3),
                "relevance_score": round(relevance_score, 3),
                "completeness_score": round(completeness_score, 3),
                "source_utilization": round(source_utilization, 3),
                "query_length": query_length,
                "response_length": response_length,
                "sources_count": sources_count
            }
            
        except Exception as e:
            logger.warning(f"⚠️ Metrics calculation failed: {e}")
            return {
                "overall_score": 0.0,
                "relevance_score": 0.0,
                "completeness_score": 0.0,
                "source_utilization": 0.0,
                "error": str(e)
            }
    
    def _fallback_evaluation(self, query: str, response: str, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Fallback evaluation when Phoenix OTEL is not available."""
        logger.info(f"Using fallback evaluation for query: {query[:50]}...")
        
        metrics = self._calculate_metrics(query, response, sources)
        
        return {
            "trace_id": str(uuid.uuid4()),
            "score": metrics.get("overall_score", 0.0),
            "metrics": metrics,
            "phoenix_logged": False,
            "fallback": True
        }
    
    def _save_trace_data(self, query: str, response: str, sources: List[Dict[str, Any]], 
                        trace_id: str, processing_time: float, evaluation_metrics: Dict[str, Any]):
        """Save trace data to files for backup and analysis."""
        try:
            # Save to JSON log
            self._save_trace_to_log(query, response, sources, trace_id, processing_time, evaluation_metrics)
            
            # Save to CSV
            self._save_trace_to_csv(query, response, sources, trace_id, processing_time, evaluation_metrics)
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to save trace data: {e}")
    
    def _save_trace_to_log(self, query: str, response: str, sources: List[Dict[str, Any]], 
                          trace_id: str, processing_time: float, evaluation_metrics: Dict[str, Any]):
        """Save trace data to a JSON log file."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = f"phoenix_otel_trace_{timestamp}_{trace_id[:8]}.json"
            log_path = os.path.join(self.log_dir, log_filename)
            
            # Create trace data
            trace_data = {
                "timestamp": datetime.now().isoformat(),
                "trace_id": trace_id,
                "query": query,
                "response": response,
                "response_length": len(response),
                "sources_count": len(sources),
                "processing_time": processing_time,
                "model": "llama2:7b",
                "framework": "llamaindex",
                "evaluation_type": "phoenix_otel",
                "evaluation_metrics": evaluation_metrics,
                "phoenix_logged": True,
                "otel_span_created": True
            }
            
            with open(log_path, 'w', encoding='utf-8') as f:
                json.dump(trace_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"📝 Phoenix OTEL trace saved to log: {log_path}")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to save trace to log: {e}")
    
    def _save_trace_to_csv(self, query: str, response: str, sources: List[Dict[str, Any]], 
                          trace_id: str, processing_time: float, evaluation_metrics: Dict[str, Any]):
        """Save trace data to CSV file."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d")
            csv_filename = f"phoenix_otel_traces_{timestamp}.csv"
            csv_path = os.path.join(self.csv_dir, csv_filename)
            
            # Prepare CSV data
            csv_data = {
                "timestamp": datetime.now().isoformat(),
                "trace_id": trace_id,
                "query": query,
                "response_length": len(response),
                "sources_count": len(sources),
                "processing_time": processing_time,
                "model": "llama2:7b",
                "framework": "llamaindex",
                "evaluation_type": "phoenix_otel",
                "overall_score": evaluation_metrics.get("overall_score", 0.0),
                "relevance_score": evaluation_metrics.get("relevance_score", 0.0),
                "completeness_score": evaluation_metrics.get("completeness_score", 0.0),
                "source_utilization": evaluation_metrics.get("source_utilization", 0.0),
                "query_length": evaluation_metrics.get("query_length", 0),
                "response_length_metrics": evaluation_metrics.get("response_length", 0),
                "sources_count_metrics": evaluation_metrics.get("sources_count", 0),
                "phoenix_logged": True,
                "otel_span_created": True
            }
            
            # Check if file exists to determine if we need to write headers
            file_exists = os.path.exists(csv_path)
            
            with open(csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=csv_data.keys())
                
                if not file_exists:
                    writer.writeheader()
                
                writer.writerow(csv_data)
            
            logger.info(f"📊 Phoenix OTEL trace saved to CSV: {csv_path}")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to save trace to CSV: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get Phoenix evaluator status."""
        # Count log and CSV files
        log_count = len([f for f in os.listdir(self.log_dir) if f.endswith('.json')]) if os.path.exists(self.log_dir) else 0
        csv_count = len([f for f in os.listdir(self.csv_dir) if f.endswith('.csv')]) if os.path.exists(self.csv_dir) else 0
        
        return {
            "phoenix_available": self.phoenix_available,
            "tracer_initialized": self.tracer is not None,
            "project_name": self.project_name,
            "log_directory": self.log_dir,
            "csv_directory": self.csv_dir,
            "log_files_count": log_count,
            "csv_files_count": csv_count,
            "phoenix_collector_endpoint": os.getenv('PHOENIX_COLLECTOR_ENDPOINT', 'http://localhost:6006'),
            "capabilities": [
                "response_evaluation",
                "query_tracing",
                "metrics_calculation",
                "otel_integration",
                "span_creation",
                "file_logging",
                "csv_export"
            ] if self.phoenix_available else ["fallback_evaluation"]
        } 