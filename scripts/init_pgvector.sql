-- Initialize pgvector extension for PostgreSQL
-- This script is automatically executed when the PostgreSQL container starts

-- Create the vector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Verify the extension is installed
SELECT vector_version();

-- Create a test table to verify vector functionality
CREATE TABLE IF NOT EXISTS test_vectors (
    id SERIAL PRIMARY KEY,
    embedding vector(384)
);

-- Insert a test vector
INSERT INTO test_vectors (embedding) VALUES ('[0.1, 0.2, 0.3]'::vector);

-- Clean up test table
DROP TABLE test_vectors;

-- Log successful initialization
DO $$
BEGIN
    RAISE NOTICE 'pgvector extension initialized successfully';
END $$; 