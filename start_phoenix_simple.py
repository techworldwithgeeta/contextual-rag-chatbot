#!/usr/bin/env python3
"""
Simple Phoenix Server Starter
"""

import phoenix as px
import time

def main():
    print("🐦 Starting Phoenix Server...")
    
    try:
        # Start Phoenix
        session = px.launch_app(port=6007)
        
        print("✅ Phoenix started successfully!")
        print("🌐 Phoenix URL: http://localhost:6007")
        print("📊 Phoenix is ready for RAG tracing!")
        
        # Keep running
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Stopping Phoenix...")
            print("✅ Phoenix stopped")
            
    except Exception as e:
        print(f"❌ Failed to start Phoenix: {e}")

if __name__ == "__main__":
    main() 