# 🤖 Contextual RAG Chatbot with Open WebUI Integration

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![RAG Pipeline](https://img.shields.io/badge/RAG-LlamaIndex%20%2B%20PGVector-green.svg)](https://github.com/run-llama/llama_index)
[![Evaluation](https://img.shields.io/badge/Evaluation-RAGAs%20%2B%20Phoenix-orange.svg)](https://github.com/explodinggradients/ragas)

A comprehensive **Procurement & Business Process Management RAG Chatbot** system that processes procurement documents, HR bylaws, and business process manuals to provide intelligent, context-aware responses. Built with modern AI/ML technologies and designed specifically for enterprise procurement and business process assistance.

## 🎯 Use Case: Procurement & Business Process Management

This RAG chatbot is specifically designed for **enterprise procurement and business process management**, providing intelligent assistance with:

### 📋 **Procurement Management**
- **Procurement Standards & Frameworks**: Abu Dhabi Procurement Standards, Ariba-aligned processes
- **Business Process Guidance**: Step-by-step procurement procedures and workflows
- **Compliance & Regulations**: Legal requirements and regulatory compliance
- **Supplier Management**: Vendor evaluation, selection, and relationship management
- **Contract Management**: Contract terms, evaluation criteria, and legal adherence

### 🏢 **Business Process Support**
- **HR Bylaws & Policies**: Human resources policies and procedures
- **Information Security**: Security standards and compliance requirements
- **Process Documentation**: Business process manuals and operational guidelines
- **Stakeholder Coordination**: Cross-functional collaboration and communication

### 💼 **Enterprise Features**
- **Multi-Document Knowledge Base**: Processes PDFs, DOCX, and text documents
- **Context-Aware Responses**: Maintains document structure and professional terminology
- **Source Citations**: Transparent reference tracking with page numbers and document sources
- **Quality Assurance**: Comprehensive evaluation and monitoring capabilities

## 🚀 Technical Features

### 📄 **Document Processing - Docling Data Pipeline & Storage**
- **Multi-format Support**: PDF, DOCX, TXT, PPTX with advanced parsing
- **Intelligent Text Extraction**: Using Docling for enterprise-grade document processing
- **Smart Chunking**: Context-aware document segmentation with metadata preservation
- **Fast Processing Mode**: Optimized for speed with OCR disabled for faster processing
- **Storage Pipeline**: Automatic document storage and vector indexing

### 🔍 **LlamaIndex + PGVector/PostgreSQL RAG Methodology**
- **Vector Search**: PGVector-powered semantic similarity search
- **PostgreSQL Integration**: Full database integration with connection pooling
- **Contextual Retrieval**: Advanced context-aware document retrieval
- **Source Citations**: Transparent reference tracking with metadata
- **Scalable Architecture**: Enterprise-ready vector database solution

### 🧠 **Contextual RAG (Anthropic-style) Embedding / LLM / Re-ranking Models**
- **Advanced Embeddings**: OpenAI text-embedding-ada-002 (1536 dimensions)
- **Contextual Re-ranking**: Multi-factor scoring (relevance, freshness, authority)
- **Intelligent Scoring**: Sophisticated document ranking algorithms
- **OpenAI Integration**: Complete embedding and re-ranking pipeline
- **Fallback Systems**: Graceful degradation when services unavailable

### 🏠 **Locally Hosted Models via Ollama**
- **Local LLM Hosting**: Complete Ollama integration for privacy and control
- **Model Support**: Gemma3:1b, Llama2:7b, Llama2:13b with dynamic switching
- **Intelligent Timeout**: Dynamic timeout management based on query complexity
- **Fallback Models**: Multiple model fallback system for reliability
- **Performance Optimization**: Efficient local model utilization

### 🤖 **Agentic Framework - Crew.AI Prompt Optimization**
- **Multi-Agent System**: Specialized agents for different tasks
- **Response Enhancement**: Advanced query processing and response improvement
- **Prompt Engineering**: A/B testing and optimization capabilities
- **Task Management**: Sequential and parallel task processing
- **Domain Expertise**: Procurement and document-specific agent specialization

### 📊 **Arize Phoenix Prompt Playground Evaluation**
- **OpenTelemetry Integration**: Complete tracing and monitoring setup
- **Query Tracing**: Comprehensive query and response tracing
- **Performance Metrics**: Real-time performance and quality metrics
- **Project Management**: Phoenix project creation and management
- **Visualization**: Interactive dashboards for prompt evaluation

### 🔬 **RAGAs LLMOps (Tracing & Debugging)**
- **Comprehensive Evaluation**: Precision, recall, faithfulness, answer correctness
- **CSV Export**: Evaluation data export and analysis capabilities
- **Fallback Metrics**: Graceful degradation when RAGAs unavailable
- **Performance Monitoring**: Real-time system health and quality tracking
- **Debugging Tools**: Advanced debugging and troubleshooting capabilities

### 🌐 **Arize Phoenix Chatbot Interface**
- **FastAPI Backend**: High-performance REST API implementation
- **Real-time Chat**: Seamless conversation experience with streaming
- **Health Monitoring**: Comprehensive system health and status endpoints
- **CORS Support**: Cross-origin request handling for web integration
- **API Documentation**: Interactive API documentation with Swagger UI

### 🔗 **Open WebUI Integration**
- **Seamless Integration**: Complete Open WebUI connector implementation
- **Model Configuration**: Automatic RAG model configuration for Open WebUI
- **API Endpoints**: Proper API endpoint configuration and management
- **Status Monitoring**: Real-time integration status and health monitoring

## 🏗️ Architecture

```mermaid
graph TB
    A[Document Upload] --> B[Docling Processor]
    B --> C[Text Extraction & Chunking]
    C --> D[Vector Embeddings]
    D --> E[PGVector Database]
    
    F[User Query] --> G[Query Processing]
    G --> H[Semantic Search]
    H --> E
    E --> I[Context Retrieval]
    I --> J[LLM Generation]
    J --> K[Response with Sources]
    
    L[Crew.AI Agents] --> M[Task Optimization]
    M --> N[Prompt Engineering]
    
    O[Arize Phoenix] --> P[Evaluation & Tracing]
    P --> Q[RAGAs Metrics]
    
    R[Open WebUI] --> S[FastAPI Backend]
    S --> T[Chat Interface]
```

## 🛠️ Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Document Processing** | Docling | Advanced document parsing and extraction |
| **RAG Framework** | LlamaIndex + PGVector | Vector search and retrieval |
| **Embedding Models** | Sentence Transformers | Contextual embeddings |
| **LLM** | Ollama (Local) | Response generation |
| **Agentic Framework** | Crew.AI | Multi-agent orchestration |
| **Evaluation** | Arize Phoenix + RAGAs | Performance metrics and tracing |
| **Web Interface** | Open WebUI + FastAPI | User interface and API |
| **Database** | PostgreSQL + pgvector | Vector storage |

## 📋 Prerequisites

### System Requirements
- **Python**: 3.12 or higher
- **PostgreSQL**: 15+ with pgvector extension
- **Ollama**: For local LLM hosting
- **UV**: Modern Python package manager
- **RAM**: Minimum 8GB (16GB recommended)
- **Storage**: 10GB+ for models and data

### Required Models (via Ollama)
```bash
# Core models for RAG pipeline
ollama pull llama2:13b
ollama pull llama2:7b


# Optional: Specialized models
ollama pull llama2:13b-chat
ollama pull codellama:13b
```

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/techworldwithgeeta/contextual-rag-chatbot.git
cd contextual-rag-chatbot
```

### 2. Install Dependencies
```bash
# Install UV if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install project dependencies
uv sync
```

### 3. Set Up PostgreSQL with pgvector
```bash
# Install PostgreSQL (Ubuntu/Debian)
sudo apt update
sudo apt install postgresql postgresql-contrib

# Install pgvector extension
# using Docker
docker pull pgvector/pgvector:pg16

docker volume create pgvector-data
command to check 
docker volume ls

docker run --name pgvector-container -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=password -e POSTGRES_DB=vectordb -p 5433:5432 -v pgvector-data:/var/lib/postgresql/data -d pgvector/pgvector:pg16


#pgadmin    
docker pull dpage/pgadmin4

docker run --name pgadmin-container -p 5050:80 -e PGADMIN_DEFAULT_EMAIL=user@domain.com -e PGADMIN_DEFAULT_PASSWORD=password -d dpage/pgadmin4
docker ps

http://localhost:5050
#provide email and password and then register server as vecordatabase and hostname as 172.17.0.3 ,username postgres


docker inspect -f '{{range.NetworkSettings.Networks}}{{.IPAddress}}{{end}}' 

# After Register server ,create database vectordb
# then query tool and enable vector extenston

CREATE EXTENSION vector;
# Set Up Open Web UI


```

### 4. Configure Environment
```bash
# Copy environment template
cp .env.example .env

# Edit with your settings
nano .env
```

**Environment Variables:**
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

# Arize Phoenix (Optional)
ARIZE_API_KEY=your_arize_api_key
ARIZE_SPACE_KEY=your_space_key

# Crew.AI (Optional)
OPENAI_API_KEY=your_openai_api_key
```

### 5. Initialize Database
```bash
# Create database schema
python scripts/init_database.py
```

### 6. Process Documents
```bash
# Add your documents to the Case Study/ folder
# Then process them
python scripts/process_documents.py
```

### 7. Start the Application
```bash
# Start the RAG backend
python main.py

# Or use the simple start script
python simple_start.py
```

### 8. Access the Interface
- **Open WebUI**: http://localhost:3000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## 📖 Usage Guide

### Basic Chat Interface
1. Open your browser and navigate to `http://localhost:3000`
2. Start a new conversation
3. Ask questions about your uploaded documents
4. View source citations and confidence scores

### API Endpoints
```bash
# Health check
curl http://localhost:8000/health

# Chat endpoint
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What are the procurement standards?"}'

# List documents
curl http://localhost:8000/documents

# Upload new document
curl -X POST http://localhost:8000/documents/upload \
  -F "file=@your_document.pdf"
```

### Advanced Features

#### Document Upload
```python
import requests

# Upload a document
with open('document.pdf', 'rb') as f:
    response = requests.post(
        'http://localhost:8001/documents/upload',
        files={'file': f}
    )
    print(response.json())
```

#### Custom RAG Queries
```python
from src.rag_pipeline.llamaindex_rag_engine import LlamaIndexRAGEngine

# Initialize RAG engine
rag_engine = LlamaIndexRAGEngine()

# Custom query with parameters
response = rag_engine.query(
    "What are the security requirements?",
    similarity_top_k=5,
    response_mode="compact"
)
```

## 🔧 Configuration

### Model Configuration
Edit `rag_chatbot_model_config.json`:
```json
{
  "llm_model": "llama2:13b",
  "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
  "chunk_size": 512,
  "chunk_overlap": 50,
  "similarity_top_k": 5,
  "response_mode": "compact"
}
```

### Performance Tuning
```python
# In your configuration
{
  "batch_size": 32,
  "max_concurrent_requests": 10,
  "cache_embeddings": true,
  "enable_reranking": true
}
```

## 🧪 Testing

### Run Test Suite
```bash
# Install test dependencies
uv sync --extra dev

# Run all tests
pytest tests/

# Run specific test categories
pytest tests/test_rag_pipeline.py
pytest tests/test_document_processing.py
pytest tests/test_evaluation.py
```

### Performance Testing
```bash
# Test RAG pipeline performance
python tests/test_performance.py

# Benchmark embedding generation
python tests/test_embeddings.py
```

## 📊 Evaluation & Monitoring

### RAGAs Evaluation
```bash
# Run RAGAs evaluation
python -m src.evaluation.ragas_evaluator

# Generate evaluation report
python export_ragas_data.py
```

### Phoenix Tracing
```bash
# Start Phoenix server
python start_phoenix_server.py

# View traces at http://localhost:6006
```

## 🐳 Docker Deployment

### Using Docker Compose
```bash
# Build and run with Docker Compose
docker-compose up -d

# View logs
docker-compose logs -f
```

### Manual Docker Build
```bash
# Build image
docker build -t contextual-rag-chatbot .

# Run container
docker run -d \
  --name rag-chatbot \
  -p 8000:8000 \
  -e DATABASE_URL=postgresql://user:pass@host:5432/db \
  contextual-rag-chatbot
```

## 🏗️ Project Structure

```
contextual-rag-chatbot/
├── src/                          # Main source code
│   ├── document_processing/      # Document processing pipeline
│   ├── rag_pipeline/            # RAG implementation
│   ├── agents/                  # Crew.AI agents
│   ├── evaluation/              # Arize Phoenix & RAGAs
│   ├── web_interface/           # FastAPI web interface
│   └── config/                  # Configuration management
├── scripts/                     # Setup and utility scripts
├── tests/                       # Test suite
├── docs/                        # Documentation
├── data/                        # Data storage (gitignored)
├── Case Study/                  # Sample documents (gitignored)
├── main.py                      # Application entry point
├── pyproject.toml              # Project configuration
└── README.md                   # This file
```

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Commit** your changes: `git commit -m 'Add amazing feature'`
4. **Push** to the branch: `git push origin feature/amazing-feature`
5. **Open** a Pull Request

### Development Setup
```bash
# Install development dependencies
uv sync --extra dev

# Set up pre-commit hooks
pre-commit install

# Run code formatting
black src/
flake8 src/
mypy src/
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support & Troubleshooting

### Common Issues

#### PostgreSQL Connection Issues
```bash
# Check PostgreSQL status
sudo systemctl status postgresql

# Verify pgvector extension
psql -U rag_user -d rag_db -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

#### Ollama Model Issues
```bash
# Check Ollama status
ollama list

# Restart Ollama service
ollama serve
```

#### Memory Issues
- Reduce `batch_size` in configuration
- Use smaller embedding models
- Increase system RAM or use swap

### Getting Help
- 📖 **Documentation**: Check the `docs/` folder
- 🐛 **Issues**: Create an issue on GitHub
- 💬 **Discussions**: Use GitHub Discussions
- 📧 **Email**: team@example.com

## 🙏 Acknowledgments

- [LlamaIndex](https://github.com/run-llama/llama_index) for the RAG framework
- [Crew.AI](https://github.com/joaomdmoura/crewAI) for agentic capabilities
- [Arize Phoenix](https://github.com/Arize-ai/phoenix) for evaluation tools
- [RAGAs](https://github.com/explodinggradients/ragas) for RAG evaluation
- [Open WebUI](https://github.com/open-webui/open-webui) for the chat interface

---

**Made with ❤️ by the RAG Chatbot Team**

*If you find this project helpful, please give it a ⭐ on GitHub!* 

