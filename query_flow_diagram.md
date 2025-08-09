# RAG System Query Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                FRONTEND (script.js)                             │
├─────────────────────────────────────────────────────────────────────────────────┤
│  1. User Input                                                                  │
│     • User types query or clicks suggestion                                    │
│     • Input disabled, loading message shown                                    │
│                                                                                 │
│  2. HTTP Request                                                                │
│     POST /api/query                                                             │
│     { query: "user question", session_id: "abc123" }                          │
│                                         │                                       │
│                                         ▼                                       │
└─────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          │
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              BACKEND API (app.py)                              │
├─────────────────────────────────────────────────────────────────────────────────┤
│  3. FastAPI Endpoint                                                            │
│     • query_documents() receives QueryRequest                                  │
│     • Creates session if needed                                                │
│     • Validates input                                                          │
│                                         │                                       │
│                                         ▼                                       │
│  4. Call RAG System                                                             │
│     rag_system.query(query, session_id)                                        │
└─────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          │
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            RAG ORCHESTRATOR (rag_system.py)                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│  5. Query Processing                                                            │
│     • Builds prompt: "Answer this question about course materials: {query}"   │
│     • Retrieves conversation history from session                              │
│     • Prepares tool definitions                                                │
│                                         │                                       │
│                                         ▼                                       │
│  6. AI Generation Call                                                          │
│     ai_generator.generate_response(query, history, tools, tool_manager)        │
└─────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          │
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           AI GENERATOR (ai_generator.py)                       │
├─────────────────────────────────────────────────────────────────────────────────┤
│  7. Claude API Call                                                             │
│     • Sends query with system prompt and tool definitions                      │
│     • System prompt guides tool usage decisions                                │
│     • Claude analyzes query type                                               │
│                                         │                                       │
│                                         ▼                                       │
│  8. Tool Decision                                                               │
│     IF course-specific question:                                                │
│     │                                                                           │
│     ├─► Claude calls search_course_content tool ─┐                            │
│     │                                             │                            │
│     └─► Direct answer (general knowledge) ───────┼─► 12. Response Generation  │
│                                                   │                            │
└───────────────────────────────────────────────────┼────────────────────────────┘
                                                    │
                                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          SEARCH TOOLS (search_tools.py)                        │
├─────────────────────────────────────────────────────────────────────────────────┤
│  9. Tool Execution                                                              │
│     • CourseSearchTool.execute(query, course_name?, lesson_number?)           │
│     • Parses search parameters                                                 │
│     • Validates filters                                                        │
│                                         │                                       │
│                                         ▼                                       │
│  10. Vector Search                                                              │
│      vector_store.search(query, course_name, lesson_number)                   │
└─────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          │
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           VECTOR STORE (vector_store.py)                       │
├─────────────────────────────────────────────────────────────────────────────────┤
│  11. Semantic Search                                                            │
│      • Embeds query using Sentence Transformers                               │
│      • Searches ChromaDB with similarity matching                             │
│      • Applies course/lesson filters                                          │
│      • Returns SearchResults with documents + metadata                        │
│                                         │                                       │
│                                         ▼                                       │
│      ┌─────────────────┐                                                       │
│      │    ChromaDB     │ ◄── Semantic similarity search                       │
│      │   (Embeddings)  │                                                       │
│      │                 │ ──► Relevant chunks + metadata                       │
│      └─────────────────┘                                                       │
└─────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          │ Search Results
                                          │
┌─────────────────────────────────────────┼───────────────────────────────────────┐
│                    RESPONSE ASSEMBLY    ▼                                       │
├─────────────────────────────────────────────────────────────────────────────────┤
│  12. Claude Response Generation (ai_generator.py)                              │
│      • Receives formatted search results                                       │
│      • Synthesizes information into coherent answer                           │
│      • No meta-commentary about search process                                │
│                                         │                                       │
│  13. Source Tracking (search_tools.py) │                                       │
│      • Extracts course titles + lesson numbers                                │
│      • Stores in last_sources for UI display                                  │
│                                         │                                       │
│  14. Session Management (rag_system.py)│                                       │
│      • Updates conversation history                                            │
│      • Returns (response, sources)                                            │
│                                         │                                       │
│  15. API Response (app.py)              │                                       │
│      • Wraps in QueryResponse model                                           │
│      • Returns JSON: {answer, sources, session_id}                           │
└─────────────────────────────────────────┼───────────────────────────────────────┘
                                          │
                                          │ HTTP Response
                                          │
┌─────────────────────────────────────────┼───────────────────────────────────────┐
│                     FRONTEND DISPLAY    ▼                                       │
├─────────────────────────────────────────────────────────────────────────────────┤
│  16. Response Handling (script.js)                                             │
│      • Removes loading message                                                 │
│      • Renders response with markdown parsing                                 │
│      • Displays sources in collapsible section                               │
│      • Re-enables input for next query                                        │
│      • Updates session_id if new session                                      │
└─────────────────────────────────────────────────────────────────────────────────┘

KEY COMPONENTS:
├── Frontend: HTML/CSS/JS interface
├── FastAPI: REST API endpoints  
├── RAG System: Query orchestration
├── AI Generator: Claude API integration
├── Search Tools: Tool definitions & execution
├── Vector Store: ChromaDB wrapper with semantic search
└── ChromaDB: Vector database with course content embeddings

DATA FLOW:
User Query → FastAPI → RAG System → Claude AI → Search Tool → Vector Store → ChromaDB
                                                    ↓
Response ← JSON API ← Sources + Answer ← Claude Synthesis ← Search Results ← Embeddings
```