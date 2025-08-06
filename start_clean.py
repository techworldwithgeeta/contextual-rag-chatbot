#!/usr/bin/env python3
"""
Clean RAG Chatbot Starter - No LangChain Issues
"""

import subprocess
import time
import sys
import os

def start_clean():
    """Start Phoenix and RAG API without any import issues."""
    print("🚀 Starting Clean RAG Chatbot...")
    
    # Start Phoenix
    print("🐦 Starting Phoenix...")
    try:
        phoenix_cmd = [
            sys.executable, "-c", 
            "import phoenix as px; print('Phoenix starting...'); px.launch_app(port=6007)"
        ]
        phoenix_process = subprocess.Popen(phoenix_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("✅ Phoenix started")
    except Exception as e:
        print(f"❌ Phoenix failed: {e}")
        return False
    
    # Wait for Phoenix
    time.sleep(5)
    
    # Start RAG API (without RAGAs)
    print("🌐 Starting RAG API...")
    try:
        api_cmd = [
            sys.executable, "-c",
            """
import sys
sys.path.append('src')
from web_interface.api import api
import uvicorn
print('RAG API starting...')
uvicorn.run(api, host='localhost', port=8001)
            """
        ]
        api_process = subprocess.Popen(api_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("✅ RAG API started")
    except Exception as e:
        print(f"❌ RAG API failed: {e}")
        return False
    
    # Wait for API
    time.sleep(8)
    
    print("\n🎉 CLEAN RAG CHATBOT STARTED!")
    print("=" * 50)
    print("🐦 Phoenix Dashboard: http://localhost:6007")
    print("🌐 RAG API Docs: http://localhost:8001/docs")
    print("💬 Chat Interface: http://localhost:8001/")
    print("=" * 50)
    print("Press Ctrl+C to stop")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopping services...")
        phoenix_process.terminate()
        api_process.terminate()
        print("✅ Services stopped")

if __name__ == "__main__":
    start_clean() 