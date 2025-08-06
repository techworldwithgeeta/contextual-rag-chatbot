#!/usr/bin/env python3
"""
Start All RAG System Services
"""

import subprocess
import time
import sys
import os
from pathlib import Path

def start_rag_api():
    """Start the RAG API server."""
    print("🚀 Starting RAG API server...")
    try:
        # Start RAG API in background
        process = subprocess.Popen([
            sys.executable, "main.py", "--no-test", "--no-docs", 
            "--host", "localhost", "--port", "8001"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a moment for startup
        time.sleep(3)
        
        if process.poll() is None:
            print("✅ RAG API server started successfully")
            return process
        else:
            print("❌ RAG API server failed to start")
            return None
    except Exception as e:
        print(f"❌ Error starting RAG API: {e}")
        return None

def start_phoenix_server():
    """Start the Phoenix server."""
    print("🐦 Starting Phoenix server...")
    try:
        # Start Phoenix in background
        process = subprocess.Popen([
            sys.executable, "start_phoenix_server.py"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a moment for startup
        time.sleep(5)
        
        if process.poll() is None:
            print("✅ Phoenix server started successfully")
            return process
        else:
            print("❌ Phoenix server failed to start")
            return None
    except Exception as e:
        print(f"❌ Error starting Phoenix: {e}")
        return None

def check_service(url, name):
    """Check if a service is running."""
    import requests
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            print(f"✅ {name} is running at {url}")
            return True
        else:
            print(f"❌ {name} check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException:
        print(f"❌ {name} not accessible at {url}")
        return False

def main():
    """Main function to start all services."""
    print("🎉 Starting All RAG System Services")
    print("="*50)
    
    processes = []
    
    # Start RAG API
    rag_process = start_rag_api()
    if rag_process:
        processes.append(("RAG API", rag_process))
    
    # Start Phoenix
    phoenix_process = start_phoenix_server()
    if phoenix_process:
        processes.append(("Phoenix", phoenix_process))
    
    # Wait for services to start
    print("\n⏳ Waiting for services to start...")
    time.sleep(10)
    
    # Check service status
    print("\n📊 Service Status Check:")
    print("-" * 30)
    
    services = [
        ("http://localhost:8001/health", "RAG API"),
        ("http://localhost:6007", "Phoenix Server"),
        ("http://localhost:3000", "Open WebUI")
    ]
    
    working_services = 0
    for url, name in services:
        if check_service(url, name):
            working_services += 1
    
    print(f"\n🎯 Status: {working_services}/{len(services)} services running")
    
    if working_services == len(services):
        print("\n🎉 All services are running!")
        print("\n📋 Access URLs:")
        print("   🌐 Open WebUI: http://localhost:3000")
        print("   📊 Phoenix Dashboard: http://localhost:6007")
        print("   🔧 RAG API: http://localhost:8001/chat")
        print("\n💡 Keep this terminal open to keep services running")
        print("   Press Ctrl+C to stop all services")
        
        try:
            # Keep services running
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Stopping all services...")
            for name, process in processes:
                try:
                    process.terminate()
                    print(f"   Stopped {name}")
                except:
                    pass
            print("✅ All services stopped")
    else:
        print(f"\n⚠️  {len(services) - working_services} services need attention")
        print("   Check the logs above for error messages")

if __name__ == "__main__":
    main() 