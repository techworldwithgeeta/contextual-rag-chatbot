#!/usr/bin/env python3
"""
Database Initialization Script for the Contextual RAG Chatbot.

This script sets up the PostgreSQL database with pgvector extension
and creates the necessary tables for the RAG system.
"""

import logging
import sys
from pathlib import Path

# Add src to Python path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.config.settings import settings
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

# Configure logging
logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL))
logger = logging.getLogger(__name__)

def create_database():
    """
    Create the PostgreSQL database if it doesn't exist.
    
    Connects to PostgreSQL and creates the database for the RAG system.
    """
    try:
        # Connect to PostgreSQL server (not to a specific database)
        # conn = psycopg2.connect(
        #     host="localhost",#"172.17.0.3", #settings.POSTGRES_HOST,
        #     port=5432,#settings.POSTGRES_PORT,
        #     user="postgres",#settings.POSTGRES_USER,
        #     password="admin",#   settings.POSTGRES_PASSWORD,
        #     database="vectordb"  # Connect to default postgres database
        # )

        conn = psycopg2.connect(
        dbname="vectordb",
        user="postgres",
        password="password",
        host="localhost",
        port="5433"
    )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        # Check if database exists
        cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (settings.POSTGRES_DB,))
        exists = cursor.fetchone()
        
        if not exists:
            # Create database
            cursor.execute(f"CREATE DATABASE {settings.POSTGRES_DB}")
            logger.info(f"Created database: {settings.POSTGRES_DB}")
        else:
            logger.info(f"Database already exists: {settings.POSTGRES_DB}")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        logger.error(f"Error creating database: {e}")
        raise

def install_pgvector_extension():
    """
    Install the pgvector extension in the database.
    
    Connects to the RAG database and installs the pgvector extension
    for vector similarity search capabilities.
    """
    try:
        # Connect to the RAG database
        # conn = psycopg2.connect(
        #     host=settings.POSTGRES_HOST,
        #     port=settings.POSTGRES_PORT,
        #     user=settings.POSTGRES_USER,
        #     password=settings.POSTGRES_PASSWORD,
        #     database=settings.POSTGRES_DB
        # )

        conn = psycopg2.connect(
        dbname="vectordb",
        user="postgres",
        password="password",
        host="localhost",
        port="5433"
    )
        cursor = conn.cursor()
        
        # Check if pgvector extension is available
        cursor.execute("SELECT 1 FROM pg_available_extensions WHERE name = 'vector'")
        available = cursor.fetchone()
        
        if not available:
            logger.error("pgvector extension is not available. Please install it first.")
            logger.error("Installation instructions: https://github.com/pgvector/pgvector")
            raise Exception("pgvector extension not available")
        
        # Check if extension is already installed
        cursor.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
        installed = cursor.fetchone()
        
        if not installed:
            # Install pgvector extension
            cursor.execute("CREATE EXTENSION vector")
            conn.commit()
            logger.info("Installed pgvector extension")
        else:
            logger.info("pgvector extension already installed")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        logger.error(f"Error installing pgvector extension: {e}")
        raise

def create_tables():
    """
    Create the necessary tables for the RAG system.
    
    Creates tables for document metadata and other system components.
    """
    try:
        # Connect to the RAG database
        # conn = psycopg2.connect(
        #     host=settings.POSTGRES_HOST,
        #     port=settings.POSTGRES_PORT,
        #     user=settings.POSTGRES_USER,
        #     password=settings.POSTGRES_PASSWORD,
        #     database=settings.POSTGRES_DB
        # )
        
    #     conn = psycopg2.connect(
    #     host="localhost",#"172.17.0.3", #settings.POSTGRES_HOST,
    #     port=5432,#settings.POSTGRES_PORT,
    #     user="postgres",#settings.POSTGRES_USER,
    #     password="admin",#   settings.POSTGRES_PASSWORD,
    #     database="vectordb"  # Connect to default postgres database
    # )

        conn = psycopg2.connect(
        dbname="vectordb",
        user="postgres",
        password="password",
        host="localhost",
        port="5433"
    )    
        cursor = conn.cursor()
        
        # Create document metadata table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS document_metadata (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                document_hash VARCHAR(64) UNIQUE NOT NULL,
                file_name VARCHAR(255) NOT NULL,
                file_path TEXT NOT NULL,
                file_type VARCHAR(10) NOT NULL,
                file_size BIGINT NOT NULL,
                total_chunks INTEGER NOT NULL,
                document_metadata JSONB,
                processing_timestamp TIMESTAMP NOT NULL,
                text_length INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create system logs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_logs (
                id SERIAL PRIMARY KEY,
                level VARCHAR(10) NOT NULL,
                message TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                component VARCHAR(50),
                metadata JSONB
            )
        """)
        
        # Create evaluation results table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS evaluation_results (
                id SERIAL PRIMARY KEY,
                evaluation_type VARCHAR(50) NOT NULL,
                query TEXT NOT NULL,
                response TEXT NOT NULL,
                metrics JSONB NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                model_used VARCHAR(100),
                processing_time FLOAT
            )
        """)
        
        # Create indexes for better performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_document_metadata_hash ON document_metadata(document_hash)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_document_metadata_timestamp ON document_metadata(processing_timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_system_logs_timestamp ON system_logs(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_evaluation_results_timestamp ON evaluation_results(timestamp)")
        
        conn.commit()
        logger.info("Created database tables and indexes")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        logger.error(f"Error creating tables: {e}")
        raise

def test_connection():
    """
    Test the database connection and basic functionality.
    
    Performs basic tests to ensure the database is working correctly.
    """
    try:
        # Connect to the database
        # conn = psycopg2.connect(
        #     host=settings.POSTGRES_HOST,
        #     port=settings.POSTGRES_PORT,
        #     user=settings.POSTGRES_USER,
        #     password=settings.POSTGRES_PASSWORD,
        #     database=settings.POSTGRES_DB
        # )
        # conn = psycopg2.connect(
        # host="localhost",#"172.17.0.3", #settings.POSTGRES_HOST,
        # port=5432,#settings.POSTGRES_PORT,
        # user="postgres",#settings.POSTGRES_USER,
        # password="admin",#   settings.POSTGRES_PASSWORD,
        # database="vectordb"  # Connect to default postgres database
        # )
         
        conn = psycopg2.connect(
        dbname="vectordb",
        user="postgres",
        password="password",
        host="localhost",
        port="5433"
    )
        cursor = conn.cursor()
        
        # Test basic query
        cursor.execute("SELECT version()")
        version = cursor.fetchone()
        logger.info(f"Connected to PostgreSQL: {version[0]}")
        
        # Test pgvector functionality
        # cursor.execute("SELECT vector_version()")
        # vector_version = cursor.fetchone()
        # logger.info(f"pgvector version: {vector_version[0]}")
        
        # Test table creation
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS test_vector (
                id SERIAL PRIMARY KEY,
                embedding vector(384)
            )
        """)
        conn.commit()
        logger.info("Vector table creation test: PASSED")
        
        # Clean up test table
        cursor.execute("DROP TABLE test_vector")
        conn.commit()
        
        cursor.close()
        conn.close()
        
        logger.info("Database connection test: PASSED")
        return True
        
    except Exception as e:
        logger.error(f"Database connection test failed: {e}")
        return False

def main():
    """
    Main function to initialize the database.
    
    Performs all necessary database setup steps.
    """
    logger.info("Starting database initialization...")
    
    try:
        # Step 1: Create database
        logger.info("Step 1: Creating database...")
        create_database()
        
        # Step 2: Install pgvector extension
        logger.info("Step 2: Installing pgvector extension...")
        install_pgvector_extension()
        
        # Step 3: Create tables
        logger.info("Step 3: Creating tables...")
        create_tables()
        
        # Step 4: Test connection
        logger.info("Step 4: Testing connection...")
        if test_connection():
            logger.info("Database initialization completed successfully!")
        else:
            logger.error("Database initialization failed!")
            sys.exit(1)
        
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 