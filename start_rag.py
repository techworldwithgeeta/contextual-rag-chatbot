#!/usr/bin/env python3
"""
Start RAG API Server
"""

from src.web_interface.api import api
import uvicorn

def main():
    print("🌐 Starting RAG API Server...")
    print("📚 API Docs will be available at: http://localhost:8001/docs")
    print("💬 Chat interface at: http://localhost:8001/")
    print("Press Ctrl+C to stop")
    
    try:
        uvicorn.run(api, host='localhost', port=8001)
    except KeyboardInterrupt:
        print("\n🛑 RAG API stopped")

if __name__ == "__main__":
    main() 