#!/usr/bin/env python3
"""
Start Phoenix Server
"""

import phoenix as px
import time

def main():
    print("🐦 Starting Phoenix Server...")
    print("🌐 Phoenix will be available at: http://localhost:6007")
    print("Press Ctrl+C to stop")
    
    try:
        # Try different ways to start Phoenix
        try:
            session = px.launch_app(port=6006)
            print("✅ Phoenix started successfully!")
        except Exception as e:
            print(f"⚠️ First method failed: {e}")
            # Try alternative method
            session = px.launch_app()
            print("✅ Phoenix started with default settings!")
        
        # Keep running
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Phoenix stopped")

if __name__ == "__main__":
    main() 