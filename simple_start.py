#!/usr/bin/env python3
"""
Simple RAG Chatbot Starter
Starts Phoenix and RAG API without complex imports
"""

import subprocess
import time
import sys
import os

def start_services():
    """Start Phoenix and RAG API services."""
    print("🚀 Starting RAG Chatbot Services...")
    
    # Start Phoenix in background
    print("🐦 Starting Phoenix...")
    try:
        phoenix_process = subprocess.Popen([
            sys.executable, "-c", 
            "import phoenix as px; px.launch_app(port=6007)"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("✅ Phoenix started")
    except Exception as e:
        print(f"❌ Phoenix failed: {e}")
        return False
    
    # Wait a moment for Phoenix to start
    time.sleep(3)
    
    # Start RAG API in background
    print("🌐 Starting RAG API...")
    try:
        api_process = subprocess.Popen([
            sys.executable, "-c",
            "from src.web_interface.api import api; import uvicorn; uvicorn.run(api, host='localhost', port=8001)"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("✅ RAG API started")
    except Exception as e:
        print(f"❌ RAG API failed: {e}")
        return False
    
    # Wait for API to start
    time.sleep(5)
    
    print("\n🎉 SERVICES STARTED SUCCESSFULLY!")
    print("=" * 50)
    print("🐦 Phoenix Dashboard: http://localhost:6007")
    print("🌐 RAG API Docs: http://localhost:8001/docs")
    print("💬 Chat Interface: http://localhost:8001/")
    print("=" * 50)
    print("Press Ctrl+C to stop all services")
    
    try:
        # Keep running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopping services...")
        phoenix_process.terminate()
        api_process.terminate()
        print("✅ Services stopped")

if __name__ == "__main__":
    start_services() 