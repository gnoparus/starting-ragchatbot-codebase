# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Running the Application
```bash
# Quick start (recommended)
./run.sh

# Manual start 
cd backend && uv run uvicorn app:app --reload --port 8000
```

### Package Management
```bash
# Install dependencies
uv sync

# Add new dependency
uv add package-name
```

### Environment Setup
Create `.env` file in root with:
```
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

## Architecture Overview

This is a **Retrieval-Augmented Generation (RAG) system** for course materials with the following architecture:

### Core Components

**Backend (`/backend/`)** - FastAPI application with modular RAG pipeline:
- `app.py` - FastAPI server with CORS middleware, serves static frontend
- `rag_system.py` - Main orchestrator coordinating all components
- `ai_generator.py` - Anthropic Claude API integration with tool calling
- `search_tools.py` - Tool management system for Claude function calling
- `vector_store.py` - ChromaDB wrapper with semantic search capabilities  
- `document_processor.py` - Course document parsing and chunking
- `session_manager.py` - Conversation history management
- `models.py` - Pydantic data models (Course, Lesson, CourseChunk)
- `config.py` - Configuration settings loaded from environment

**Frontend (`/frontend/`)** - Static web interface:
- `index.html` - Chat interface with course statistics
- `script.js` - Handles user input, API calls, and response rendering
- `style.css` - UI styling

### Data Flow Architecture

1. **Document Processing**: Course files parsed into structured format with metadata extraction
2. **Vector Storage**: Content chunked and embedded using Sentence Transformers, stored in ChromaDB
3. **Query Processing**: User queries processed through FastAPI → RAG System → Claude AI
4. **Tool Calling**: Claude uses search tools to find relevant course content when needed
5. **Response Assembly**: Search results synthesized into coherent answers with source attribution

### Key Design Patterns

**Tool-Based RAG**: Uses Anthropic's tool calling instead of traditional retrieve-then-generate:
- Claude decides when to search based on query analysis
- `CourseSearchTool` handles semantic search with course/lesson filtering
- Sources tracked automatically for UI display

**Modular Components**: Each major function separated into dedicated modules:
- `VectorStore` abstracts ChromaDB operations
- `DocumentProcessor` handles all text parsing and chunking
- `AIGenerator` manages Claude API interactions
- `ToolManager` coordinates tool definitions and execution

**Session Management**: Maintains conversation context across queries using session IDs

### Document Format

Course files expected structure:
```
Course Title: [title]
Course Link: [url] 
Course Instructor: [instructor]

Lesson 1: [lesson title]
Lesson Link: [optional lesson url]
[lesson content...]

Lesson 2: [lesson title]
[lesson content...]
```

### Configuration

Key settings in `config.py`:
- `CHUNK_SIZE: 800` - Text chunk size for embeddings
- `CHUNK_OVERLAP: 100` - Overlap between chunks for context
- `MAX_RESULTS: 5` - Maximum search results returned
- `MAX_HISTORY: 2` - Conversation messages to remember
- `EMBEDDING_MODEL: "all-MiniLM-L6-v2"` - Sentence transformer model
- `ANTHROPIC_MODEL: "claude-sonnet-4-20250514"` - Claude model version

### API Endpoints

- `POST /api/query` - Process user queries with optional session context
- `GET /api/courses` - Retrieve course statistics and metadata
- `GET /` - Serve frontend static files

## Development Notes

### Adding New Course Content
Place `.txt` files in `/docs/` directory. The system automatically loads them on startup and processes according to the expected document format.

### Vector Store Operations  
ChromaDB persisted at `./backend/chroma_db/`. The system handles:
- Automatic course metadata extraction and storage
- Deduplication to prevent re-processing existing courses
- Semantic search with course and lesson filtering

### Tool System Extension
To add new tools:
1. Implement `Tool` interface in `search_tools.py`
2. Define tool schema for Claude function calling
3. Register with `ToolManager` in `rag_system.py`

### Session Management
Sessions auto-created on first query, maintain conversation history for context. Configure history length via `MAX_HISTORY` setting.
- always use uv to run the server do not use pip directly
- make sure to use uv to manage all dependencies
- use uv to run python files