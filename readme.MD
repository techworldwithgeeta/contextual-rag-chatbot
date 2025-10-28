# 🤖 Contextual RAG Chatbot with Open WebUI Integration

**Author: Geeta khatwani**

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![RAG Pipeline](https://img.shields.io/badge/RAG-LlamaIndex%20%2B%20PGVector-green.svg)](https://github.com/run-llama/llama_index)
[![Evaluation](https://img.shields.io/badge/Evaluation-RAGAs%20%2B%20Phoenix-orange.svg)](https://github.com/explodinggradients/ragas)
[![Document Processing](https://img.shields.io/badge/Document%20Processing-Docling-blue.svg)](https://github.com/DS4SD/docling)
[![Agentic Framework](https://img.shields.io/badge/Agentic%20Framework-Crew.AI-purple.svg)](https://github.com/joaomdmoura/crewAI)

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
    C --> D[OpenAI Embeddings]
    D --> E[PGVector Database]
    
    F[User Query] --> G[Query Processing]
    G --> H[Semantic Search]
    H --> E
    E --> I[Context Retrieval]
    I --> J[Contextual Re-ranking]
    J --> K[Ollama LLM Generation]
    K --> L[Crew.AI Enhancement]
    L --> M[Response with Sources]
    
    N[Crew.AI Agents] --> O[Task Optimization]
    O --> P[Prompt Engineering]
    
    Q[Arize Phoenix] --> R[Evaluation & Tracing]
    R --> S[RAGAs Metrics]
    
    T[Open WebUI] --> U[FastAPI Backend]
    U --> V[Chat Interface]
    
    W[Phoenix Server] --> X[OTEL Tracing]
    X --> Y[Performance Monitoring]
```

## 🛠️ Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Document Processing** | Docling | Advanced document parsing and extraction |
| **RAG Framework** | LlamaIndex + PGVector | Vector search and retrieval |
| **Embedding Models** | OpenAI text-embedding-ada-002 | Contextual embeddings (1536 dimensions) |
| **LLM** | Ollama (Local) | Response generation with multiple models |
| **Re-ranking** | OpenAI + Custom Algorithms | Intelligent document re-ranking |
| **Agentic Framework** | Crew.AI | Multi-agent orchestration and optimization |
| **Evaluation** | Arize Phoenix + RAGAs | Performance metrics and tracing |
| **Web Interface** | FastAPI + Open WebUI | User interface and API |
| **Database** | PostgreSQL + pgvector | Vector storage and management |
| **Tracing** | OpenTelemetry | Comprehensive system monitoring |

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
ollama pull gemma3:1b
ollama pull llama2:7b
ollama pull llama2:13b

# Optional: Specialized models
ollama pull mistral:7b
ollama pull codellama:7b
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
# Using Docker (Recommended)
docker pull pgvector/pgvector:pg16
docker volume create pgvector-data
docker run --name pgvector-container \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=password \
  -e POSTGRES_DB=vectordb \
  -p 5433:5432 \
  -v pgvector-data:/var/lib/postgresql/data \
  -d pgvector/pgvector:pg16

# Enable pgvector extension
docker exec -it pgvector-container psql -U postgres -d vectordb -c "CREATE EXTENSION vector;"
```

### 4. Set Up Ollama
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama service
ollama serve

# Pull required models
ollama pull gemma3:1b
ollama pull llama2:7b
```

### 5. Configure Environment
```bash
# Copy environment template
cp env.example .env

# Edit with your settings
nano .env
```

**Environment Variables:**
```env
# Database Configuration
DATABASE_URL=postgresql://postgres:password@localhost:5433/vectordb
POSTGRES_HOST=localhost
POSTGRES_PORT=5433
POSTGRES_DB=vectordb
POSTGRES_USER=postgres
POSTGRES_PASSWORD=password

# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gemma3:1b
OLLAMA_TIMEOUT=60

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here
EMBEDDING_MODEL=text-embedding-ada-002
EMBEDDING_DIMENSION=1536

# Application Settings
WEB_HOST=localhost
WEB_PORT=8001
LOG_LEVEL=INFO
FAST_DOCUMENT_PROCESSING=true

# Phoenix Configuration (Optional)
PHOENIX_PROJECT_NAME=rag-chatbot
PHOENIX_COLLECTOR_ENDPOINT=http://localhost:6006
ENABLE_PHOENIX_TRACING=true

# RAGAs Configuration (Optional)
ENABLE_RAGAS_EVALUATION=true
RAGAS_DATASET_PATH=data/evaluation/
```

### 6. Start the Application

**Option 1: Start Phoenix Server (Recommended)**
```bash
# Terminal 1: Start Phoenix server
python start_phoenix_server.py

# Terminal 2: Start main application
python main.py --host localhost --port 8001
```

**Option 2: Start without Phoenix**
```bash
# Start main application only
python main.py --host localhost --port 8001 --no-phoenix
```

### 7. Access the Interface
- **API Documentation**: http://localhost:8001/docs
- **Health Check**: http://localhost:8001/health
- **Chat Endpoint**: http://localhost:8001/chat
- **Phoenix Dashboard**: http://localhost:6006 (if Phoenix is running)

## 📖 Usage Guide

### Basic Chat Interface
1. Add your procurement documents to the `Case Study/` folder:
   - **Procurement Standards**: Abu Dhabi Procurement Standards, Ariba-aligned manuals
   - **HR Documents**: HR Bylaws and policies
   - **Security Documents**: Information Security standards
   - **Process Manuals**: Business process documentation
2. The system will automatically process them on startup
3. Use the API endpoints to interact with the chatbot
4. View source citations and confidence scores in responses

### Example Queries
```bash
# Procurement Standards
"What are the procurement standards?"
"How does the procurement process work?"
"What are the key principles of procurement?"

# Business Process Management
"Explain Section 4: Procurement Clients and Partners"
"What documents are required for procurement?"
"How is procurement transparency ensured?"

# HR and Compliance
"What are the HR bylaws?"
"What are the information security requirements?"
```

### API Endpoints
```bash
# Health check
curl http://localhost:8001/health

# Chat endpoint - Procurement query
curl -X POST http://localhost:8001/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the procurement standards?"}'

# Chat endpoint - Business process query
curl -X POST http://localhost:8001/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Explain Section 4: Procurement Clients and Partners"}'

# List documents
curl http://localhost:8001/documents

# System statistics
curl http://localhost:8001/stats
```

### Advanced Features

#### Document Processing
```python
# Documents are automatically processed from Case Study/ folder
# Supported formats: PDF, DOCX, TXT
# Processing includes: text extraction, chunking, embedding generation
```

#### Custom RAG Queries
```python
from src.rag_pipeline.llamaindex_rag_engine import LlamaIndexRAGEngine

# The RAG engine is automatically initialized with:
# - OpenAI embeddings
# - Ollama LLM
# - PGVector storage
# - Crew.AI agents
# - Phoenix tracing
# - RAGAs evaluation
```

## 🔧 Configuration

### Model Configuration
The system uses the following default models:
- **LLM**: Gemma3:1b (via Ollama)
- **Embeddings**: text-embedding-ada-002 (OpenAI)
- **Re-ranking**: text-embedding-3-large (OpenAI)

### Performance Tuning
```env
# In your .env file
OLLAMA_TIMEOUT=60
FAST_DOCUMENT_PROCESSING=true
VECTOR_STORE_TOP_K=5
ENABLE_RERANKING=true
```

## 🧪 Testing

### Run System Tests
```bash
# Test the complete system
python main.py --no-docs --test-only

# Test specific components
python -c "from src.rag_pipeline.llamaindex_rag_engine import LlamaIndexRAGEngine; print('RAG Engine OK')"
```

### Procurement-Specific Testing
```bash
# Test API endpoints with procurement queries
curl http://localhost:8001/health
curl -X POST http://localhost:8001/chat -H "Content-Type: application/json" -d '{"query": "What are the procurement standards?"}'
curl -X POST http://localhost:8001/chat -H "Content-Type: application/json" -d '{"query": "How does the procurement process work?"}'

# Test guard rails
curl http://localhost:8001/test/guard_rails
```

## 💼 Business Value & Use Cases

### 🎯 **Primary Use Cases**
- **Procurement Teams**: Quick access to procurement standards, procedures, and compliance requirements
- **Business Process Managers**: Guidance on operational procedures and stakeholder coordination
- **HR Professionals**: Reference for HR bylaws and policy information
- **Compliance Officers**: Information security standards and regulatory requirements
- **New Employees**: Onboarding assistance with company policies and procedures

### 📈 **Key Benefits**
- **Reduced Response Time**: Instant access to complex procurement procedures
- **Consistency**: Standardized responses based on official documentation
- **Compliance**: Ensures responses align with official policies and standards
- **Efficiency**: Eliminates need to manually search through multiple documents
- **Accuracy**: Source citations provide transparency and verification

### 🔍 **Evaluation Metrics**
Based on our evaluation data, the system demonstrates:
- **Response Length Score**: 94.2% (comprehensive responses)
- **Keyword Relevance**: 43.9% (good semantic matching)
- **Faithfulness**: 40.2% (accurate to source documents)
- **Answer Correctness**: 34.0% (overall quality score)
- **Success Rate**: 100% (no system failures)

## 📊 Evaluation & Monitoring

### Phoenix Tracing
```bash
# Start Phoenix server
python start_phoenix_server.py

# View traces at http://localhost:6006
# Traces include: query processing, response generation, evaluation metrics
```

### RAGAs Evaluation
```bash
# Evaluation runs automatically during query processing
# Results are exported to data/evaluation/ directory
# CSV files are generated for analysis
```

## 🏗️ Project Structure

```
contextual-rag-chatbot/
├── src/                          # Main source code
│   ├── document_processing/      # Docling document processing pipeline
│   │   └── processor.py         # Document processor with fast mode
│   ├── rag_pipeline/            # RAG implementation
│   │   ├── llamaindex_rag_engine.py  # Main RAG engine
│   │   ├── vector_store.py      # PGVector integration
│   │   └── reranker.py          # Contextual re-ranking
│   ├── agents/                  # Crew.AI agents
│   │   └── crew_agents.py       # Multi-agent system
│   ├── evaluation/              # Evaluation frameworks
│   │   ├── phoenix_evaluator.py # Arize Phoenix integration
│   │   └── ragas_evaluator.py   # RAGAs evaluation
│   ├── web_interface/           # FastAPI web interface
│   │   ├── api.py              # Main API endpoints
│   │   └── open_webui_connector.py # Open WebUI integration
│   └── config/                  # Configuration management
│       └── settings.py          # Application settings
├── Case Study/                  # Procurement & business documents (auto-processed)
│   ├── Abu Dhabi Procurement Standards.PDF
│   ├── Procurement Manual (Ariba Aligned).PDF
│   ├── Procurement Manual (Business Process).PDF
│   ├── HR Bylaws.pdf
│   └── Information Security.pdf
├── data/                        # Data storage (gitignored)
│   ├── processed_documents/     # Processed document chunks
│   ├── vector_cache/           # Embedding cache
│   └── evaluation/             # Evaluation results
├── logs/                        # Application logs
├── main.py                      # Application entry point
├── start_phoenix_server.py     # Phoenix server launcher
├── setup_phoenix_otel.py       # Phoenix OTEL setup
├── pyproject.toml              # Project configuration
├── env.example                 # Environment template
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
# Check if PostgreSQL is running
docker ps | grep pgvector

# Check connection
docker exec -it pgvector-container psql -U postgres -d vectordb -c "SELECT version();"
```

#### Ollama Model Issues
```bash
# Check Ollama status
ollama list

# Restart Ollama service
ollama serve

# Pull missing models
ollama pull gemma3:1b
```

#### Memory Issues
- Reduce `OLLAMA_TIMEOUT` in environment
- Use smaller models (gemma3:1b instead of llama2:13b)
- Enable `FAST_DOCUMENT_PROCESSING=true`

### Getting Help
- 📖 **Documentation**: Check the `SETUP_GUIDE.md` file
- 🐛 **Issues**: Create an issue on GitHub
- 💬 **Discussions**: Use GitHub Discussions
- 📧 **Email**: team@example.com

## 🙏 Acknowledgments

- [LlamaIndex](https://github.com/run-llama/llama_index) for the RAG framework
- [Crew.AI](https://github.com/joaomdmoura/crewAI) for agentic capabilities
- [Arize Phoenix](https://github.com/Arize-ai/phoenix) for evaluation tools
- [RAGAs](https://github.com/explodinggradients/ragas) for RAG evaluation
- [Docling](https://github.com/DS4SD/docling) for document processing
- [Open WebUI](https://github.com/open-webui/open-webui) for the chat interface

---

**Made with ❤️ by Geeta khatwani**

*If you find this project helpful, please give it a ⭐ on GitHub!*
