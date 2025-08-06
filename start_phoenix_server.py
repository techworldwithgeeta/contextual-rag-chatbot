#!/usr/bin/env python3
"""
Persistent Phoenix Server for RAG Tracing
"""

import phoenix as px
import time
import webbrowser
import signal
import sys

def signal_handler(signum, frame):
    """Handle shutdown signals."""
    print("\n🛑 Shutting down Phoenix server...")
    sys.exit(0)

def start_phoenix_server():
    """Start Phoenix server and keep it running."""
    print("🐦 Starting Phoenix Server for RAG Tracing...")
    
    try:
        # Set environment variable for port
        import os
        os.environ['PHOENIX_PORT'] = '6006'
        
        # Start Phoenix with proper configuration
        try:
            session = px.launch_app(
                port=6006,
                run_in_thread=True
            )
        except TypeError:
            # Try alternative method
            session = px.launch_app(
                run_in_thread=True
            )
        
        print(f"✅ Phoenix started successfully!")
        print(f"🌐 Phoenix URL: {session.url}")
        
        # Open browser
        webbrowser.open(session.url)
        
        print("\n📊 Phoenix is now ready for RAG tracing!")
        print("📋 What you'll see:")
        print("   - Traces: All RAG queries with performance data")
        print("   - Evaluations: Response quality metrics")
        print("   - Spans: Detailed processing steps")
        print("\n🚀 Phoenix server is running. Keep this terminal open.")
        print("📊 Open http://localhost:6006 to view traces")
        print("🔄 Phoenix will automatically receive traces from your RAG system")
        
        # Keep the server running
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Phoenix server stopped by user")
    except Exception as e:
        print(f"❌ Failed to start Phoenix: {e}")

if __name__ == "__main__":
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    start_phoenix_server() 