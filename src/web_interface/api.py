"""
RAG Chatbot API using FastAPI.
"""

import logging
import os
import glob
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import time
import uuid

from src.config.settings import settings

logger = logging.getLogger(__name__)

class RAGChatbotAPI:
    """RAG Chatbot API using FastAPI."""
    
    def __init__(self):
        """Initialize the API."""
        self.app = FastAPI(
            title="Contextual RAG Chatbot API",
            description="A RAG chatbot with Phoenix tracing and evaluation",
            version="1.0.0"
        )
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Initialize RAG engine (will be set by main.py)
        self.rag_engine = None
        
        # Register routes
        self._register_routes()
    
    def set_rag_engine(self, rag_engine):
        """Set the RAG engine instance."""
        self.rag_engine = rag_engine
        logger.info("RAG engine set in API")
        
    def _register_routes(self):
        """Register API routes."""
        
        @self.app.get("/")
        async def root():
            """Root endpoint."""
            return {"message": "Contextual RAG Chatbot API", "status": "running"}
        
        @self.app.get("/health")
        async def health():
            """Health check endpoint."""
            return {"status": "healthy", "service": "rag_chatbot"}
        
        @self.app.get("/status")
        async def status():
            """Comprehensive status endpoint showing all component status."""
            try:
                status_info = {
                    "service": "rag_chatbot",
                    "timestamp": time.time(),
                    "components": {
                        "api_server": {
                            "status": "running",
                            "url": f"http://localhost:8001",
                            "endpoints": {
                                "health": "/health",
                                "status": "/status", 
                                "chat": "/chat",
                                "chat_completions": "/chat/completions",
                                "docs": "/docs"
                            }
                        },
                        "rag_engine": {
                            "status": "available" if self.rag_engine else "unavailable",
                            "framework": "llamaindex" if self.rag_engine else None,
                            "model": self.rag_engine.get_status() if self.rag_engine else None
                        },
                        "crew_ai": {
                            "status": "available" if (self.rag_engine and self.rag_engine.crew_manager) else "unavailable",
                            "agents": self.rag_engine.crew_manager.get_status() if (self.rag_engine and self.rag_engine.crew_manager) else None
                        },
                        "phoenix": {
                            "status": "available" if (self.rag_engine and self.rag_engine.phoenix_evaluator) else "check_manually",
                            "url": "http://localhost:6007",
                            "description": "Phoenix tracing and evaluation dashboard"
                        },
                        "open_webui": {
                            "status": "check_manually", 
                            "url": "http://localhost:3000",
                            "description": "Open WebUI chat interface",
                            "setup_required": True
                        },
                        "vector_store": {
                            "status": "available" if (self.rag_engine and self.rag_engine.vector_store_manager and self.rag_engine.vector_store_manager.vector_store) else "unavailable",
                            "type": "SimpleVectorStore" if (self.rag_engine and self.rag_engine.vector_store_manager and self.rag_engine.vector_store_manager.vector_store) else None,
                            "description": "In-memory vector store for document storage"
                        },
                        "embeddings": {
                            "status": "available" if (self.rag_engine and hasattr(self.rag_engine, 'embedding_model')) else "unavailable",
                            "model": "text-embedding-ada-002" if (self.rag_engine and hasattr(self.rag_engine, 'embedding_model')) else None
                        }
                    },
                    "configuration": {
                        "ollama_model": "llama2:7b",
                        "embedding_model": "text-embedding-ada-002",
                        "crew_ai_llm": "gpt-3.5-turbo" if (self.rag_engine and self.rag_engine.crew_manager) else "unavailable"
                    },
                    "system_status": {
                        "overall": "healthy",
                        "crew_ai_working": True if (self.rag_engine and self.rag_engine.crew_manager) else False,
                        "phoenix_working": True if (self.rag_engine and self.rag_engine.phoenix_evaluator) else False,
                        "api_responding": True
                    }
                }
                
                # Add detailed RAG engine status if available
                if self.rag_engine:
                    rag_status = self.rag_engine.get_status()
                    status_info["components"]["rag_engine"].update(rag_status)
                
                return status_info
                
            except Exception as e:
                logger.error(f"Error in status endpoint: {e}")
                return {
                    "service": "rag_chatbot",
                    "status": "error",
                    "error": str(e),
                    "timestamp": time.time()
                }
        
        @self.app.post("/chat")
        async def chat(request: Dict[str, Any]):
            """Chat endpoint."""
            try:
                query = request.get("query", "")
                if not query:
                    raise HTTPException(status_code=400, detail="Query is required")
                
                # Use the actual RAG engine if available
                if self.rag_engine:
                    try:
                        logger.info(f"Processing query: {query}")
                        rag_response = self.rag_engine.query(query)
                        
                        if rag_response and "response" in rag_response:
                            response_text = rag_response["response"]
                            sources = rag_response.get("source_documents", [])
                            processing_time = rag_response.get("processing_time", 0.1)
                            logger.info(f"RAG response generated: {len(response_text)} characters")
                        else:
                            response_text = f"RAG Response: {query}"
                            sources = []
                            processing_time = 0.1
                            logger.warning("RAG engine returned empty response")
                            
                    except Exception as e:
                        logger.error(f"Error calling RAG engine: {e}")
                        response_text = f"RAG Response: {query} (Error: {str(e)})"
                        sources = []
                        processing_time = 0.1
                else:
                    logger.warning("RAG engine not available, using fallback response")
                    response_text = f"RAG Response: {query}"
                    sources = []
                    processing_time = 0.1
                
                response = {
                    "response": response_text,
                    "sources": sources,
                    "processing_time": processing_time
                }
                
                return response
                
            except Exception as e:
                logger.error(f"Error in chat endpoint: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/export/csv")
        async def export_csv(request: Dict[str, Any] = None):
            """Export evaluation data to CSV."""
            try:
                if self.rag_engine and hasattr(self.rag_engine, 'ragas_evaluator') and self.rag_engine.ragas_evaluator:
                    # Get export options from request
                    simplified = True  # Default to simplified
                    if request and 'simplified' in request:
                        simplified = request['simplified']
                    
                    # Export evaluation data
                    detailed_csv = self.rag_engine.ragas_evaluator.export_evaluation_data(simplified=simplified)
                    
                    # Export summary data
                    summary_csv = None
                    if self.rag_engine.ragas_evaluator.generate_evaluation_summary_csv():
                        # Get the latest summary file
                        summary_files = glob.glob("data/evaluation/ragas_evaluation_summary_*.csv")
                        if summary_files:
                            summary_csv = max(summary_files, key=os.path.getctime)
                    
                    return {
                        "status": "success",
                        "message": f"CSV files exported successfully ({'simplified' if simplified else 'full'} format)",
                        "files": {
                            "detailed_data": detailed_csv,
                            "summary_data": summary_csv
                        },
                        "format": "simplified" if simplified else "full"
                    }
                else:
                    raise HTTPException(status_code=400, detail="RAGAs evaluator not available")
                    
            except Exception as e:
                logger.error(f"Error exporting CSV: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/chat/completions")
        async def chat_completions(request: Dict[str, Any]):
            """OpenAI-compatible chat completions endpoint for Open WebUI."""
            try:
                # Extract messages from request
                messages = request.get("messages", [])
                if not messages:
                    raise HTTPException(status_code=400, detail="Messages are required")
                
                # Get the last user message
                user_message = None
                for msg in reversed(messages):
                    if msg.get("role") == "user":
                        user_message = msg.get("content", "")
                        break
                
                if not user_message:
                    raise HTTPException(status_code=400, detail="User message not found")
                
                # Get other parameters
                model = request.get("model", "gpt-3.5-turbo")
                temperature = request.get("temperature", 0.7)
                max_tokens = request.get("max_tokens", 2048)
                
                logger.info(f"Processing chat completion: {user_message[:50]}...")
                
                # Use the actual RAG engine if available
                if self.rag_engine:
                    try:
                        logger.info("Calling RAG engine with Crew.AI...")
                        rag_response = self.rag_engine.query(user_message)
                        
                        if rag_response and "response" in rag_response:
                            response_text = rag_response["response"]
                            logger.info(f"RAG response generated: {len(response_text)} characters")
                        else:
                            response_text = f"RAG Response: {user_message}"
                            logger.warning("RAG engine returned empty response")
                            
                    except Exception as e:
                        logger.error(f"Error calling RAG engine: {e}")
                        response_text = f"RAG Response: {user_message} (Error: {str(e)})"
                else:
                    logger.warning("RAG engine not available, using fallback response")
                    response_text = f"RAG Response: {user_message}"
                
                # Create OpenAI-compatible response
                response = {
                    "id": f"chatcmpl-{uuid.uuid4().hex}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": response_text
                            },
                            "finish_reason": "stop"
                        }
                    ],
                    "usage": {
                        "prompt_tokens": len(user_message.split()),
                        "completion_tokens": len(response_text.split()),
                        "total_tokens": len(user_message.split()) + len(response_text.split())
                    }
                }
                
                logger.info(f"Returning response with {len(response_text)} characters")
                return response
                
            except Exception as e:
                logger.error(f"Error in chat completions endpoint: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
    def run(self, host: str = "localhost", port: int = 8001):
        """Run the API server."""
        logger.info(f"Starting RAG API server on {host}:{port}")
        uvicorn.run(self.app, host=host, port=port)

# Create global API instance
api = RAGChatbotAPI().app 