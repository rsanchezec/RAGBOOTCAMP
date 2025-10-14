# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a RAG (Retrieval-Augmented Generation) Bootcamp project focused on teaching the fundamentals of building RAG systems using LangChain. The repository is structured as a progressive learning path with Jupyter notebooks covering data ingestion, embeddings, and vector stores.

## Environment Setup

**Python Version**: 3.12 (specified in `.python-version`)

**Virtual Environment**: The project uses a `.venv` virtual environment

**Installation**:
```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Environment Variables**: The project requires API keys in `.env`:
- `OPENAI_API_KEY` - For OpenAI embeddings and models
- `GROQ_API_KEY` - For Groq API access
- `LANGSMITH_API_KEY` - Optional, for LangSmith tracing

## Repository Architecture

The codebase is organized into numbered modules representing a progressive learning path:

### Module Structure

**000_DataIngestParsing/** - Data loading and parsing techniques
- Text files (.txt) with `TextLoader` and `DirectoryLoader`
- PDF parsing with multiple methods (PyPDF, PyMuPDF, Unstructured)
- Word documents (.docx) with docx2txt
- Structured data (CSV, Excel, JSON)
- Database ingestion
- Text splitting strategies (CharacterTextSplitter, RecursiveCharacterTextSplitter, TokenTextSplitter)

**001_VectorEmbeddingAndDatabases/** - Embedding fundamentals
- Embedding concepts and cosine similarity
- HuggingFace embeddings (sentence-transformers)
- OpenAI embeddings
- Comparison of different embedding models

**003_VectorStores/** - Vector database implementations
- ChromaDB for persistent vector storage
- FAISS for efficient similarity search

### Data Directory Structure
```
000_DataIngestParsing/data/
├── text_files/     # Plain text samples
├── pdf/            # PDF documents
├── word_files/     # .docx files
├── structured_files/ # CSV/Excel files
├── json_files/     # JSON data
└── databases/      # Database files
```

## Key Technologies

**LangChain**: Core framework (v0.3) for building RAG pipelines
- Document loaders for various file formats
- Text splitters for chunking strategies
- Embedding integrations

**Embeddings**:
- HuggingFace: sentence-transformers models (all-MiniLM-L6-v2, all-mpnet-base-v2)
- OpenAI: text-embedding-ada-002 and newer models

**Vector Stores**:
- ChromaDB: Persistent vector database with metadata filtering
- FAISS: Facebook's similarity search library for in-memory operations

**Document Processing**:
- pypdf, pymupdf - PDF parsing
- python-docx, docx2txt - Word document handling
- pandas, openpyxl - Structured data (CSV/Excel)
- unstructured, pdfminer - Advanced document parsing

## Running Notebooks

This is a notebook-based project. To work with it:

```bash
# Install Jupyter if not already installed
pip install jupyter

# Launch Jupyter
jupyter notebook

# Or use VS Code with Jupyter extension
code .
```

**Notebook Execution Order**:
1. Start with `000_DataIngestParsing/1-dataingestion.ipynb` for basic concepts
2. Progress through numbered notebooks in each module sequentially
3. Each notebook builds on concepts from previous ones

## Common Development Tasks

**Running a specific notebook**:
```bash
jupyter notebook 000_DataIngestParsing/1-dataingestion.ipynb
```

**Testing embeddings**:
- Notebooks in `001_VectorEmbeddingAndDatabases/` demonstrate embedding creation
- Use HuggingFace models for local testing (no API key needed)
- Switch to OpenAI embeddings for production quality

**Working with vector stores**:
- ChromaDB persists to `003_VectorStores/chroma_db/` directory
- FAISS indexes are created in-memory but can be saved/loaded

## Architecture Patterns

**Document Loading Pattern**:
1. Load documents with appropriate loader (TextLoader, PyPDFLoader, etc.)
2. Documents have `.page_content` and `.metadata` attributes
3. Metadata is crucial for filtering, source tracking, and providing context

**Text Splitting Strategy**:
- **CharacterTextSplitter**: Simple, predictable splits on delimiters
- **RecursiveCharacterTextSplitter**: Recommended default, tries multiple separators hierarchically
- **TokenTextSplitter**: For token-aware splitting (respects model token limits)
- Common parameters: `chunk_size`, `chunk_overlap`, `separators`

**Embedding Workflow**:
1. Initialize embedding model (HuggingFace or OpenAI)
2. Use `.embed_query(text)` for single text
3. Use `.embed_documents(list)` for batch processing
4. Embeddings are numerical vectors (384-768 dimensions typically)

**Vector Store Pattern**:
1. Create embeddings from documents
2. Initialize vector store (ChromaDB/FAISS)
3. Add documents with `.add_documents(docs)`
4. Query with `.similarity_search(query, k=n)` for retrieval

## Important Notes

- All notebooks use LangChain v0.3 syntax
- The project uses `python-dotenv` to load environment variables
- Sentence-transformers models download automatically on first use (~100MB)
- ChromaDB creates persistent storage in local directories
- FAISS is optimized for speed but requires more setup for persistence

## Module Naming Convention

Modules use a numeric prefix system (000_, 001_, 003_) indicating the learning sequence. Note that module 002_ may be added or the numbering represents a course structure with some modules not yet included.
