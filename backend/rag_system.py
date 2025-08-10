from typing import List, Tuple, Optional, Dict
import os
import time
from document_processor import DocumentProcessor
from vector_store import VectorStore
from ai_generator import AIGenerator
from session_manager import SessionManager
from search_tools import ToolManager, CourseSearchTool, CourseOutlineTool
from models import Course, Lesson, CourseChunk
from logger import get_logger, log_execution_time

# Initialize logger for this module
log = get_logger("rag_system")

class RAGSystem:
    """Main orchestrator for the Retrieval-Augmented Generation system"""
    
    def __init__(self, config):
        log.info("Initializing RAG System", config_chunk_size=config.CHUNK_SIZE, config_max_results=config.MAX_RESULTS)
        
        self.config = config
        
        # Initialize core components
        log.debug("Initializing document processor")
        self.document_processor = DocumentProcessor(config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        
        log.debug("Initializing vector store", chroma_path=config.CHROMA_PATH, embedding_model=config.EMBEDDING_MODEL)
        self.vector_store = VectorStore(config.CHROMA_PATH, config.EMBEDDING_MODEL, config.MAX_RESULTS)
        
        log.debug("Initializing AI generator", anthropic_model=config.ANTHROPIC_MODEL)
        self.ai_generator = AIGenerator(config.ANTHROPIC_API_KEY, config.ANTHROPIC_MODEL)
        
        log.debug("Initializing session manager", max_history=config.MAX_HISTORY)
        self.session_manager = SessionManager(config.MAX_HISTORY)
        
        # Initialize search tools
        log.debug("Initializing search tools")
        self.tool_manager = ToolManager()
        self.search_tool = CourseSearchTool(self.vector_store)
        self.outline_tool = CourseOutlineTool(self.vector_store)
        self.tool_manager.register_tool(self.search_tool)
        self.tool_manager.register_tool(self.outline_tool)
        
        log.success("RAG System initialization completed")
    
    @log_execution_time("add_course_document")
    def add_course_document(self, file_path: str) -> Tuple[Course, int]:
        """
        Add a single course document to the knowledge base.
        
        Args:
            file_path: Path to the course document
            
        Returns:
            Tuple of (Course object, number of chunks created)
        """
        log.info(f"Processing course document: {file_path}")
        
        try:
            # Process the document
            log.debug("Processing document with document processor", file_path=file_path)
            course, course_chunks = self.document_processor.process_course_document(file_path)
            
            if not course:
                log.warning("No course data extracted from document", file_path=file_path)
                return None, 0
            
            log.debug(
                "Document processed successfully",
                file_path=file_path,
                course_title=course.title,
                chunks_created=len(course_chunks)
            )
            
            # Add course metadata to vector store for semantic search
            log.debug("Adding course metadata to vector store", course_title=course.title)
            self.vector_store.add_course_metadata(course)
            
            # Add course content chunks to vector store
            log.debug("Adding course content chunks to vector store", chunks_count=len(course_chunks))
            self.vector_store.add_course_content(course_chunks)
            
            log.success(
                "Course document added successfully",
                file_path=file_path,
                course_title=course.title,
                chunks_created=len(course_chunks)
            )
            
            return course, len(course_chunks)
        except Exception as e:
            log.error(
                "Error processing course document",
                file_path=file_path,
                error=str(e),
                error_type=type(e).__name__
            )
            return None, 0
    
    @log_execution_time("add_course_folder")
    def add_course_folder(self, folder_path: str, clear_existing: bool = False) -> Tuple[int, int]:
        """
        Add all course documents from a folder.
        
        Args:
            folder_path: Path to folder containing course documents
            clear_existing: Whether to clear existing data first
            
        Returns:
            Tuple of (total courses added, total chunks created)
        """
        log.info(f"Processing course folder: {folder_path}", clear_existing=clear_existing)
        
        total_courses = 0
        total_chunks = 0
        
        # Clear existing data if requested
        if clear_existing:
            log.warning("Clearing existing data for fresh rebuild")
            self.vector_store.clear_all_data()
        
        if not os.path.exists(folder_path):
            log.error(f"Folder does not exist: {folder_path}")
            return 0, 0
        
        # Get existing course titles to avoid re-processing
        log.debug("Retrieving existing course titles")
        existing_course_titles = set(self.vector_store.get_existing_course_titles())
        log.info(f"Found {len(existing_course_titles)} existing courses")
        
        # Get list of files to process
        all_files = [f for f in os.listdir(folder_path) 
                    if os.path.isfile(os.path.join(folder_path, f)) 
                    and f.lower().endswith(('.pdf', '.docx', '.txt'))]
        
        log.info(f"Found {len(all_files)} files to process", files=all_files)
        
        # Process each file in the folder
        for i, file_name in enumerate(all_files, 1):
            file_path = os.path.join(folder_path, file_name)
            log.debug(f"Processing file {i}/{len(all_files)}: {file_name}")
            
            try:
                # Check if this course might already exist
                # We'll process the document to get the course ID, but only add if new
                course, course_chunks = self.document_processor.process_course_document(file_path)
                
                if course and course.title not in existing_course_titles:
                    # This is a new course - add it to the vector store
                    log.debug(f"Adding new course to vector store: {course.title}")
                    self.vector_store.add_course_metadata(course)
                    self.vector_store.add_course_content(course_chunks)
                    total_courses += 1
                    total_chunks += len(course_chunks)
                    
                    log.success(
                        f"Added new course: {course.title}",
                        chunks_created=len(course_chunks),
                        progress=f"{i}/{len(all_files)}"
                    )
                    existing_course_titles.add(course.title)
                    
                elif course:
                    log.info(f"Course already exists, skipping: {course.title}")
                else:
                    log.warning(f"No course data extracted from {file_name}")
                    
            except Exception as e:
                log.error(
                    f"Error processing file: {file_name}",
                    error=str(e),
                    error_type=type(e).__name__,
                    progress=f"{i}/{len(all_files)}"
                )
        
        log.success(
            "Course folder processing completed",
            folder_path=folder_path,
            total_courses_added=total_courses,
            total_chunks_created=total_chunks,
            files_processed=len(all_files)
        )
        
        return total_courses, total_chunks
    
    @log_execution_time("query_processing")
    def query(self, query: str, session_id: Optional[str] = None) -> Tuple[str, List[str]]:
        """
        Process a user query using the RAG system with tool-based search.
        
        Args:
            query: User's question
            session_id: Optional session ID for conversation context
            
        Returns:
            Tuple of (response, sources list - empty for tool-based approach)
        """
        log.info(
            "Processing user query",
            session_id=session_id,
            query_length=len(query),
            query_preview=query[:100] + "..." if len(query) > 100 else query
        )
        
        # Create prompt for the AI with clear instructions
        prompt = f"""Answer this question about course materials: {query}"""
        
        # Get conversation history if session exists
        history = None
        if session_id:
            log.debug("Retrieving conversation history", session_id=session_id)
            history = self.session_manager.get_conversation_history(session_id)
            if history:
                log.debug(f"Found {len(history)} previous exchanges in history", session_id=session_id)
        
        # Generate response using AI with tools
        log.debug("Generating AI response with tools", available_tools=len(self.tool_manager.get_tool_definitions()))
        response = self.ai_generator.generate_response(
            query=prompt,
            conversation_history=history,
            tools=self.tool_manager.get_tool_definitions(),
            tool_manager=self.tool_manager
        )
        
        # Get sources from the search tool
        sources = self.tool_manager.get_last_sources()
        log.debug(f"Retrieved {len(sources)} sources from tool searches")

        # Reset sources after retrieving them
        self.tool_manager.reset_sources()
        
        # Update conversation history
        if session_id:
            log.debug("Updating conversation history", session_id=session_id)
            self.session_manager.add_exchange(session_id, query, response)
        
        log.success(
            "Query processing completed",
            session_id=session_id,
            response_length=len(response),
            sources_count=len(sources)
        )
        
        # Return response with sources from tool searches
        return response, sources
    
    def get_course_analytics(self) -> Dict:
        """Get analytics about the course catalog"""
        log.debug("Retrieving course analytics")
        
        try:
            total_courses = self.vector_store.get_course_count()
            course_titles = self.vector_store.get_existing_course_titles()
            
            analytics = {
                "total_courses": total_courses,
                "course_titles": course_titles
            }
            
            log.info(
                "Course analytics retrieved",
                total_courses=total_courses,
                titles_count=len(course_titles)
            )
            
            return analytics
            
        except Exception as e:
            log.error(
                "Error retrieving course analytics",
                error=str(e),
                error_type=type(e).__name__
            )
            raise