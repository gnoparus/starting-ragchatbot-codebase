import warnings
warnings.filterwarnings("ignore", message="resource_tracker: There appear to be.*")

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import time
import uuid

from config import config
from rag_system import RAGSystem
from logger import get_api_logger, get_logger

# Initialize loggers
api_logger = get_api_logger("app")
app_logger = get_logger("app")

# Initialize FastAPI app
app = FastAPI(title="Course Materials RAG System", root_path="")
app_logger.info("Initializing Course Materials RAG System")

# Add trusted host middleware for proxy
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]
)

# Enable CORS with proper settings for proxy
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Initialize RAG system
app_logger.info("Initializing RAG system with config", config={
    "chunk_size": config.CHUNK_SIZE,
    "max_results": config.MAX_RESULTS,
    "anthropic_model": config.ANTHROPIC_MODEL,
    "embedding_model": config.EMBEDDING_MODEL
})
rag_system = RAGSystem(config)
app_logger.success("RAG system initialized successfully")

# Pydantic models for request/response
class QueryRequest(BaseModel):
    """Request model for course queries"""
    query: str
    session_id: Optional[str] = None

class Source(BaseModel):
    """Model for a source with optional link"""
    text: str
    link: Optional[str] = None

class QueryResponse(BaseModel):
    """Response model for course queries"""
    answer: str
    sources: List[Source]
    session_id: str

class CourseStats(BaseModel):
    """Response model for course statistics"""
    total_courses: int
    course_titles: List[str]

# API Endpoints

@app.post("/api/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest, http_request: Request):
    """Process a query and return response with sources"""
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    
    client_ip = http_request.client.host if http_request.client else "unknown"
    
    api_logger.info(
        "API Query Request",
        request_id=request_id,
        client_ip=client_ip,
        query_length=len(request.query),
        session_id=request.session_id,
        query_preview=request.query[:100] + "..." if len(request.query) > 100 else request.query
    )
    
    try:
        # Create session if not provided
        session_id = request.session_id
        if not session_id:
            session_id = rag_system.session_manager.create_session()
            api_logger.debug("Created new session", request_id=request_id, session_id=session_id)
        
        # Process query using RAG system
        answer, sources = rag_system.query(request.query, session_id)
        
        processing_time = time.time() - start_time
        api_logger.success(
            "API Query Success",
            request_id=request_id,
            session_id=session_id,
            processing_time=f"{processing_time:.3f}s",
            answer_length=len(answer),
            sources_count=len(sources)
        )
        
        return QueryResponse(
            answer=answer,
            sources=sources,
            session_id=session_id
        )
    except Exception as e:
        processing_time = time.time() - start_time
        api_logger.error(
            "API Query Error",
            request_id=request_id,
            session_id=request.session_id,
            processing_time=f"{processing_time:.3f}s",
            error=str(e),
            error_type=type(e).__name__
        )
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/courses", response_model=CourseStats)
async def get_course_stats(http_request: Request):
    """Get course analytics and statistics"""
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    
    client_ip = http_request.client.host if http_request.client else "unknown"
    
    api_logger.info("API Course Stats Request", request_id=request_id, client_ip=client_ip)
    
    try:
        analytics = rag_system.get_course_analytics()
        
        processing_time = time.time() - start_time
        api_logger.success(
            "API Course Stats Success",
            request_id=request_id,
            processing_time=f"{processing_time:.3f}s",
            total_courses=analytics["total_courses"]
        )
        
        return CourseStats(
            total_courses=analytics["total_courses"],
            course_titles=analytics["course_titles"]
        )
    except Exception as e:
        processing_time = time.time() - start_time
        api_logger.error(
            "API Course Stats Error",
            request_id=request_id,
            processing_time=f"{processing_time:.3f}s",
            error=str(e),
            error_type=type(e).__name__
        )
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def startup_event():
    """Load initial documents on startup"""
    app_logger.info("Application startup initiated")
    
    docs_path = "../docs"
    if os.path.exists(docs_path):
        app_logger.info(f"Loading initial documents from {docs_path}")
        try:
            start_time = time.time()
            courses, chunks = rag_system.add_course_folder(docs_path, clear_existing=False)
            loading_time = time.time() - start_time
            
            app_logger.success(
                "Document loading completed",
                courses_loaded=courses,
                chunks_created=chunks,
                loading_time=f"{loading_time:.3f}s"
            )
        except Exception as e:
            app_logger.error(f"Error loading documents: {e}")
    else:
        app_logger.warning(f"Documents folder not found: {docs_path}")
    
    app_logger.success("Application startup completed")

# Custom static file handler with no-cache headers for development
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
from pathlib import Path


class DevStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        if isinstance(response, FileResponse):
            # Add no-cache headers for development
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response
    
    
# Serve static files for the frontend
app.mount("/", StaticFiles(directory="../frontend", html=True), name="static")