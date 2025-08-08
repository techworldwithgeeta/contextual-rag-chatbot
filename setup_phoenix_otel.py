#!/usr/bin/env python3
"""
Phoenix OTEL Setup Script

This script demonstrates the proper setup for Phoenix OTEL (OpenTelemetry)
following the official Phoenix documentation.
"""

import os
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_phoenix_environment():
    """Setup Phoenix environment variables."""
    logger.info("Setting up Phoenix OTEL environment...")
    
    # Set Phoenix collector endpoint
    phoenix_endpoint = os.getenv('PHOENIX_COLLECTOR_ENDPOINT', 'http://localhost:6006')
    os.environ['PHOENIX_COLLECTOR_ENDPOINT'] = phoenix_endpoint
    
    logger.info(f"✅ Phoenix collector endpoint: {phoenix_endpoint}")
    
    # Set project name
    project_name = os.getenv('PHOENIX_PROJECT_NAME', 'rag-chatbot')
    os.environ['PHOENIX_PROJECT_NAME'] = project_name
    
    logger.info(f"✅ Phoenix project name: {project_name}")
    
    return phoenix_endpoint, project_name

def setup_phoenix_otel():
    """Setup Phoenix OTEL with proper configuration."""
    try:
        # Import Phoenix OTEL
        from phoenix.otel import register
        
        logger.info("✅ Phoenix OTEL imported successfully")
        
        # Get environment variables
        phoenix_endpoint, project_name = setup_phoenix_environment()
        
        # Register the application with Phoenix OTEL
        # This should be called BEFORE any code execution
        logger.info("Registering application with Phoenix OTEL...")
        
        tracer_provider = register(
            project_name=project_name,
            endpoint=f"{phoenix_endpoint}/v1/traces",
            auto_instrument=True
        )
        
        logger.info("✅ Phoenix OTEL registration completed")
        logger.info(f"📊 Project: {project_name}")
        logger.info(f"🌐 Endpoint: {phoenix_endpoint}")
        
        return tracer_provider
        
    except ImportError as e:
        logger.error(f"❌ Phoenix OTEL not available: {e}")
        logger.info("💡 Install with: pip install arize-phoenix-otel")
        return None
    except Exception as e:
        logger.error(f"❌ Phoenix OTEL setup failed: {e}")
        return None

def demonstrate_tracing():
    """Demonstrate Phoenix OTEL tracing."""
    try:
        from opentelemetry import trace
        
        # Get the tracer
        tracer = trace.get_tracer(__name__)
        
        logger.info("✅ Tracer obtained successfully")
        
        # Create a sample span
        with tracer.start_as_current_span(
            "sample_operation",
            attributes={
                "operation_type": "setup_demo",
                "framework": "phoenix_otel"
            }
        ) as span:
            span.set_attributes({
                "demo_completed": True,
                "setup_successful": True
            })
            
            logger.info("✅ Sample span created successfully")
            
        return True
        
    except Exception as e:
        logger.error(f"❌ Tracing demonstration failed: {e}")
        return False

def main():
    """Main setup function."""
    logger.info("🚀 Phoenix OTEL Setup Script")
    logger.info("=" * 50)
    
    # Setup Phoenix OTEL
    tracer_provider = setup_phoenix_otel()
    
    if tracer_provider:
        # Demonstrate tracing
        tracing_success = demonstrate_tracing()
        
        if tracing_success:
            logger.info("\n🎉 Phoenix OTEL setup completed successfully!")
            logger.info("\n📋 Next Steps:")
            logger.info("1. Start Phoenix server: phoenix start")
            logger.info("2. Access dashboard: http://localhost:6006")
            logger.info("3. Run your RAG application")
            logger.info("4. View traces in Phoenix dashboard")
        else:
            logger.error("❌ Tracing demonstration failed")
    else:
        logger.error("❌ Phoenix OTEL setup failed")
        sys.exit(1)

if __name__ == "__main__":
    main() 