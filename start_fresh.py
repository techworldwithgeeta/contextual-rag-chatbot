#!/usr/bin/env python3
"""
Fresh RAG Chatbot Starter - Clean Cache & Start Services
"""

import subprocess
import time
import sys
import os
import shutil

def clean_cache():
    """Clean all cache directories."""
    print("🧹 Cleaning cache...")
    
    cache_dirs = [
        '.phoenix',
        '__pycache__', 
        '.pytest_cache',
        'data/cache',
        'logs',
        'temp',
        'tmp'
    ]
    
    for cache_dir in cache_dirs:
        try:
            if os.path.exists(cache_dir):
                shutil.rmtree(cache_dir)
                print(f"✅ Cleaned {cache_dir}")
        except Exception as e:
            print(f"⚠️ Could not clean {cache_dir}: {e}")
    
    # Clean all __pycache__ directories recursively
    try:
        for root, dirs, files in os.walk('.'):
            if '__pycache__' in dirs:
                cache_path = os.path.join(root, '__pycache__')
                shutil.rmtree(cache_path)
                print(f"✅ Cleaned {cache_path}")
    except Exception as e:
        print(f"⚠️ Could not clean all __pycache__: {e}")
    
    print("✅ Cache cleaning completed")

def start_services():
    """Start Phoenix and RAG API services."""
    print("🚀 Starting Fresh RAG Chatbot...")
    
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
    
    # Start RAG API
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
    
    print("\n🎉 FRESH RAG CHATBOT STARTED!")
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

def main():
    """Main function."""
    print("🧹 FRESH RAG CHATBOT STARTUP")
    print("=" * 40)
    
    # Clean cache first
    clean_cache()
    
    # Start services
    start_services()

if __name__ == "__main__":
    main() 