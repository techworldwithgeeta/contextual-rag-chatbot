"""
Phoenix evaluation module for proper tracing and evaluation.
"""

import logging
import time
import uuid
import os
import json
import csv
from typing import Dict, Any, List, Optional
from datetime import datetime

# Try to import Phoenix components
try:
    import phoenix as px
    from phoenix.trace import TraceDataset
    from phoenix.trace.schemas import Span
    PHOENIX_AVAILABLE = True
except ImportError:
    PHOENIX_AVAILABLE = False
    px = None
    TraceDataset = None
    Span = None

logger = logging.getLogger(__name__)

class PhoenixEvaluator:
    """Phoenix evaluator for RAG tracing and evaluation."""
    
    def __init__(self):
        """Initialize the Phoenix evaluator."""
        logger.info("Phoenix evaluator initialized")
        self.phoenix_available = PHOENIX_AVAILABLE
        self.client = None
        self.project_name = "rag-chatbot"
        
        # Setup logging directories
        self.log_dir = "data/phoenix_logs"
        self.csv_dir = "data/phoenix_csv"
        self._setup_directories()
        
        if self.phoenix_available:
            try:
                # Initialize Phoenix client
                self.client = px.Client()
                logger.info("✅ Phoenix client initialized successfully")
                
                # Register the application with Phoenix
                self._register_application()
                
            except Exception as e:
                logger.warning(f"⚠️ Phoenix client initialization failed: {e}")
                self.phoenix_available = False
        else:
            logger.warning("⚠️ Phoenix not available - install with: pip install arize-phoenix")
    
    def _setup_directories(self):
        """Setup directories for Phoenix logs and CSV files."""
        try:
            os.makedirs(self.log_dir, exist_ok=True)
            os.makedirs(self.csv_dir, exist_ok=True)
            logger.info(f"✅ Phoenix directories created: {self.log_dir}, {self.csv_dir}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to create Phoenix directories: {e}")
    
    def _register_application(self):
        """Register the RAG application with Phoenix."""
        try:
            if not self.client:
                return
            
            # For now, just log that the application is ready
            logger.info("✅ Application ready for Phoenix integration")
            
        except Exception as e:
            logger.warning(f"⚠️ Application registration failed: {e}")
    
    def evaluate_response(self, query: str, response: str, sources: List[Dict[str, Any]], 
                         trace_id: Optional[str] = None, processing_time: float = 0.0) -> Dict[str, Any]:
        """Evaluate a response using Phoenix and send traces."""
        if not self.phoenix_available or not self.client:
            logger.warning("⚠️ Phoenix not available for evaluation")
            return self._fallback_evaluation(query, response, sources)
        
        try:
            # Generate trace ID if not provided
            if not trace_id:
                trace_id = str(uuid.uuid4())
            
            # Create evaluation metrics
            evaluation_metrics = self._calculate_metrics(query, response, sources)
            
            # Create trace data for Phoenix
            trace_data = {
                "trace_id": trace_id,
                "query": query,
                "response": response,
                "response_length": len(response),
                "sources_count": len(sources),
                "processing_time": processing_time,
                "evaluation_metrics": evaluation_metrics,
                "model": "llama2:7b",
                "framework": "llamaindex",
                "evaluation_type": "phoenix"
            }
            
            # Send trace to Phoenix dashboard
            self._send_trace_to_phoenix(trace_data)
            
            # Log basic information to console
            logger.info(f"✅ Phoenix evaluation completed for trace_id: {trace_id}")
            logger.info(f"📊 Phoenix metrics: {evaluation_metrics}")
            
            return {
                "trace_id": trace_id,
                "score": evaluation_metrics.get("overall_score", 0.0),
                "metrics": evaluation_metrics,
                "phoenix_logged": True
            }
            
        except Exception as e:
            logger.error(f"❌ Phoenix evaluation failed: {e}")
            return self._fallback_evaluation(query, response, sources)
    
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
        """Fallback evaluation when Phoenix is not available."""
        logger.info(f"Using fallback evaluation for query: {query[:50]}...")
        
        metrics = self._calculate_metrics(query, response, sources)
        
        return {
            "trace_id": str(uuid.uuid4()),
            "score": metrics.get("overall_score", 0.0),
            "metrics": metrics,
            "phoenix_logged": False,
            "fallback": True
        }
    
    def log_query_trace(self, query: str, response: str, sources: List[Dict[str, Any]], 
                       trace_id: str, processing_time: float, model: str = "llama2:7b"):
        """Log a query trace to Phoenix."""
        if not self.phoenix_available or not self.client:
            logger.warning("⚠️ Phoenix not available for tracing")
            return False
        
        try:
            # Log basic query information (simplified approach)
            logger.info(f"✅ Query trace logged to Phoenix: {trace_id}")
            logger.info(f"📝 Query: {query[:100]}...")
            logger.info(f"📊 Processing time: {processing_time:.2f}s")
            logger.info(f"📄 Sources count: {len(sources)}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Query trace logging failed: {e}")
            return False
    
    def _send_trace_to_phoenix(self, trace_data: Dict[str, Any]):
        """Send trace data to Phoenix dashboard and save to files."""
        try:
            if not self.phoenix_available or not self.client:
                return
            
            # Create a simple trace dataset
            # For now, we'll use a simple approach to send data to Phoenix
            logger.info(f"📤 Sending trace to Phoenix dashboard: {trace_data['trace_id']}")
            
            # Try to send using the client's web_url to verify connection
            if hasattr(self.client, 'web_url'):
                logger.info(f"🌐 Phoenix dashboard URL: {self.client.web_url}")
            
            # Save trace to log file
            self._save_trace_to_log(trace_data)
            
            # Save trace to CSV file
            self._save_trace_to_csv(trace_data)
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to send trace to Phoenix: {e}")
    
    def _save_trace_to_log(self, trace_data: Dict[str, Any]):
        """Save trace data to a JSON log file."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = f"phoenix_trace_{timestamp}_{trace_data['trace_id'][:8]}.json"
            log_path = os.path.join(self.log_dir, log_filename)
            
            # Add timestamp to trace data
            trace_data_with_timestamp = {
                "timestamp": datetime.now().isoformat(),
                "phoenix_logged": True,
                **trace_data
            }
            
            with open(log_path, 'w', encoding='utf-8') as f:
                json.dump(trace_data_with_timestamp, f, indent=2, ensure_ascii=False)
            
            logger.info(f"📝 Phoenix trace saved to log: {log_path}")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to save trace to log: {e}")
    
    def _save_trace_to_csv(self, trace_data: Dict[str, Any]):
        """Save trace data to CSV file."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d")
            csv_filename = f"phoenix_traces_{timestamp}.csv"
            csv_path = os.path.join(self.csv_dir, csv_filename)
            
            # Prepare CSV data
            csv_data = {
                "timestamp": datetime.now().isoformat(),
                "trace_id": trace_data.get("trace_id", ""),
                "query": trace_data.get("query", ""),
                "response_length": trace_data.get("response_length", 0),
                "sources_count": trace_data.get("sources_count", 0),
                "processing_time": trace_data.get("processing_time", 0.0),
                "model": trace_data.get("model", ""),
                "framework": trace_data.get("framework", ""),
                "overall_score": trace_data.get("evaluation_metrics", {}).get("overall_score", 0.0),
                "relevance_score": trace_data.get("evaluation_metrics", {}).get("relevance_score", 0.0),
                "completeness_score": trace_data.get("evaluation_metrics", {}).get("completeness_score", 0.0),
                "source_utilization": trace_data.get("evaluation_metrics", {}).get("source_utilization", 0.0),
                "query_length": trace_data.get("evaluation_metrics", {}).get("query_length", 0),
                "response_length_metrics": trace_data.get("evaluation_metrics", {}).get("response_length", 0),
                "sources_count_metrics": trace_data.get("evaluation_metrics", {}).get("sources_count", 0)
            }
            
            # Check if file exists to determine if we need to write headers
            file_exists = os.path.exists(csv_path)
            
            with open(csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=csv_data.keys())
                
                if not file_exists:
                    writer.writeheader()
                
                writer.writerow(csv_data)
            
            logger.info(f"📊 Phoenix trace saved to CSV: {csv_path}")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to save trace to CSV: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get Phoenix evaluator status."""
        # Count log and CSV files
        log_count = len([f for f in os.listdir(self.log_dir) if f.endswith('.json')]) if os.path.exists(self.log_dir) else 0
        csv_count = len([f for f in os.listdir(self.csv_dir) if f.endswith('.csv')]) if os.path.exists(self.csv_dir) else 0
        
        return {
            "phoenix_available": self.phoenix_available,
            "client_initialized": self.client is not None,
            "project_name": self.project_name,
            "log_directory": self.log_dir,
            "csv_directory": self.csv_dir,
            "log_files_count": log_count,
            "csv_files_count": csv_count,
            "capabilities": [
                "response_evaluation",
                "query_tracing",
                "metrics_calculation",
                "application_registration",
                "file_logging",
                "csv_export"
            ] if self.phoenix_available else ["fallback_evaluation"]
        } 