# 🚀 Complete Setup Guide

This guide provides detailed step-by-step instructions for setting up the Contextual RAG Chatbot system.

## 📋 Prerequisites Checklist

Before starting, ensure you have:

- [ ] Python 3.12+ installed
- [ ] PostgreSQL 15+ with pgvector extension
- [ ] Ollama installed and running
- [ ] UV package manager installed
- [ ] Git installed
- [ ] At least 8GB RAM available
- [ ] 10GB+ free disk space

## 🔧 Step-by-Step Installation

### 1. System Dependencies

#### Ubuntu/Debian
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3.12
sudo apt install python3.12 python3.12-venv python3.12-dev

# Install PostgreSQL
sudo apt install postgresql postgresql-contrib

# Install pgvector extension
sudo apt install postgresql-15-pgvector

# Install build tools
sudo apt install build-essential libpq-dev
```

#### macOS
```bash
# Install Homebrew if not installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python 3.12
brew install python@3.12

# Install PostgreSQL
brew install postgresql@15

# Install pgvector
brew install pgvector
```

#### Windows
```bash
# Install Python 3.12 from python.org
# Install PostgreSQL from postgresql.org
# Install pgvector using the Windows installer
```

### 2. Install UV Package Manager

```bash
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Restart your terminal or source the profile
source ~/.bashrc  # or ~/.zshrc
```

### 3. Clone and Setup Project

```bash
# Clone the repository
git clone https://github.com/your-username/contextual-rag-chatbot.git
cd contextual-rag-chatbot

# Install dependencies
uv sync

# Activate virtual environment
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate     # Windows
```

### 4. PostgreSQL Setup

#### Create Database and User
```bash
# Connect to PostgreSQL as superuser
sudo -u postgres psql

# Create database and user
CREATE DATABASE rag_db;
CREATE USER rag_user WITH PASSWORD 'rag_password';
GRANT ALL PRIVILEGES ON DATABASE rag_db TO rag_user;
\c rag_db
CREATE EXTENSION IF NOT EXISTS vector;
\q
```

#### Test Connection
```bash
# Test database connection
psql -h localhost -U rag_user -d rag_db -c "SELECT version();"
```

### 5. Ollama Setup

#### Install Ollama
```bash
# Download and install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama service
ollama serve
```

#### Download Required Models
```bash
# Core models (this may take time depending on your internet)
ollama pull llama2:13b
ollama pull llama2:7b
ollama pull mistral:7b
ollama pull codellama:7b

# Verify models are installed
ollama list
```

### 6. Environment Configuration

#### Create Environment File
```bash
# Copy the example environment file
cp .env.example .env

# Edit the environment file
nano .env  # or use your preferred editor
```

#### Environment Variables
```env
# Database Configuration
DATABASE_URL=postgresql://rag_user:rag_password@localhost:5432/rag_db
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=rag_db
POSTGRES_USER=rag_user
POSTGRES_PASSWORD=rag_password

# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama2:13b

# Embedding Model
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Application Settings
DEBUG=False
LOG_LEVEL=INFO
HOST=0.0.0.0
PORT=8000

# Arize Phoenix (Optional - for evaluation)
ARIZE_API_KEY=your_arize_api_key
ARIZE_SPACE_KEY=your_space_key

# Crew.AI (Optional - for agentic features)
OPENAI_API_KEY=your_openai_api_key

# Security (Change these in production)
SECRET_KEY=your-secret-key-here
```

### 7. Database Initialization

```bash
# Initialize database schema
python scripts/init_database.py

# Verify tables are created
psql -h localhost -U rag_user -d rag_db -c "\dt"
```

### 8. Document Processing Setup

#### Prepare Documents
```bash
# Create documents directory
mkdir -p "Case Study"

# Add your documents to the Case Study/ folder
# Supported formats: PDF, DOCX, TXT, PPTX
```

#### Process Documents
```bash
# Process all documents in the Case Study folder
python scripts/process_documents.py

# Verify processing
ls -la data/processed_documents/
```

### 9. Start the Application

#### Development Mode
```bash
# Start the application
python main.py

# Or use the simple start script
python simple_start.py
```

#### Production Mode
```bash
# Using uvicorn directly
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4

# Or using the production script
python start_all_services.py
```

### 10. Verify Installation

#### Check Services
```bash
# Check if all services are running
curl http://localhost:8000/health

# Check Ollama
curl http://localhost:11434/api/tags

# Check PostgreSQL
psql -h localhost -U rag_user -d rag_db -c "SELECT COUNT(*) FROM documents;"
```

#### Test the Interface
1. Open browser: http://localhost:8000
2. Start a new conversation
3. Ask a question about your documents
4. Verify response quality and source citations

## 🔍 Troubleshooting

### Common Issues and Solutions

#### PostgreSQL Connection Issues
```bash
# Check PostgreSQL status
sudo systemctl status postgresql

# Restart PostgreSQL
sudo systemctl restart postgresql

# Check pgvector extension
psql -h localhost -U rag_user -d rag_db -c "SELECT * FROM pg_extension WHERE extname = 'vector';"
```

#### Ollama Issues
```bash
# Check Ollama status
ollama list

# Restart Ollama
pkill ollama
ollama serve

# Check model availability
ollama show llama2:13b
```

#### Memory Issues
```bash
# Check available memory
free -h

# Reduce batch size in configuration
# Edit rag_chatbot_model_config.json
{
  "batch_size": 16,
  "chunk_size": 256
}
```

#### Port Conflicts
```bash
# Check what's using port 8000
lsof -i :8000

# Kill conflicting process
sudo kill -9 <PID>

# Or use different port
python main.py --port 8001
```

### Performance Optimization

#### For Low-RAM Systems (< 8GB)
```bash
# Use smaller models
ollama pull llama2:7b
# Update .env: OLLAMA_MODEL=llama2:7b

# Reduce batch sizes
# Edit configuration files
```

#### For High-Performance Systems
```bash
# Use larger models
ollama pull llama2:70b

# Increase workers
uvicorn main:app --workers 8

# Enable caching
# Set cache_embeddings=true in config
```

## 🧪 Testing Your Setup

### Basic Functionality Test
```bash
# Run basic tests
python -m pytest tests/test_basic_functionality.py -v

# Test document processing
python -m pytest tests/test_document_processing.py -v

# Test RAG pipeline
python -m pytest tests/test_rag_pipeline.py -v
```

### Performance Benchmark
```bash
# Run performance tests
python tests/test_performance.py

# Check response times
python tests/test_response_times.py
```

### Evaluation Setup (Optional)
```bash
# Install evaluation dependencies
uv sync --extra evaluation

# Start Phoenix server
python start_phoenix_server.py

# Run RAGAs evaluation
python -m src.evaluation.ragas_evaluator
```

## 🚀 Next Steps

After successful setup:

1. **Customize Configuration**: Edit `rag_chatbot_model_config.json` for your use case
2. **Add More Documents**: Upload additional documents to the Case Study folder
3. **Fine-tune Models**: Adjust parameters based on your specific requirements
4. **Set Up Monitoring**: Configure logging and monitoring for production use
5. **Deploy to Production**: Use Docker or cloud deployment options

## 📞 Support

If you encounter issues:

1. Check the troubleshooting section above
2. Review the logs in the `logs/` directory
3. Create an issue on GitHub with detailed error information
4. Check the documentation in the `docs/` folder

---

**Happy RAG-ing! 🤖✨** 