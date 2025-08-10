"""Tests for RAG System functionality"""
import pytest
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from rag_system import RAGSystem
from models import Course, CourseChunk


class TestRAGSystem:
    """Test cases for RAGSystem"""

    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    def test_initialization(self, mock_session_manager, mock_ai_generator, 
                          mock_vector_store, mock_document_processor, test_config):
        """Test RAGSystem initialization"""
        rag = RAGSystem(test_config)
        
        # Check that all components were initialized
        mock_document_processor.assert_called_once_with(
            test_config.CHUNK_SIZE, 
            test_config.CHUNK_OVERLAP
        )
        mock_vector_store.assert_called_once_with(
            test_config.CHROMA_PATH, 
            test_config.EMBEDDING_MODEL, 
            test_config.MAX_RESULTS
        )
        mock_ai_generator.assert_called_once_with(
            test_config.ANTHROPIC_API_KEY, 
            test_config.ANTHROPIC_MODEL
        )
        mock_session_manager.assert_called_once_with(test_config.MAX_HISTORY)
        
        # Check that tools were registered
        assert rag.tool_manager is not None
        assert rag.search_tool is not None
        assert rag.outline_tool is not None

    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')  
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    def test_add_course_document_success(self, mock_session_manager, mock_ai_generator,
                                       mock_vector_store, mock_document_processor, 
                                       test_config, sample_course, sample_course_chunks):
        """Test successful course document addition"""
        # Setup mocks
        mock_doc_processor = Mock()
        mock_document_processor.return_value = mock_doc_processor
        mock_doc_processor.process_course_document.return_value = (sample_course, sample_course_chunks)
        
        mock_store = Mock()
        mock_vector_store.return_value = mock_store
        
        rag = RAGSystem(test_config)
        
        # Test the method
        course, chunk_count = rag.add_course_document("/path/to/course.txt")
        
        assert course == sample_course
        assert chunk_count == len(sample_course_chunks)
        
        # Check that methods were called correctly
        mock_doc_processor.process_course_document.assert_called_once_with("/path/to/course.txt")
        mock_store.add_course_metadata.assert_called_once_with(sample_course)
        mock_store.add_course_content.assert_called_once_with(sample_course_chunks)

    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    def test_add_course_document_error(self, mock_session_manager, mock_ai_generator,
                                     mock_vector_store, mock_document_processor, test_config):
        """Test course document addition with error"""
        # Setup mocks to raise exception
        mock_doc_processor = Mock()
        mock_document_processor.return_value = mock_doc_processor
        mock_doc_processor.process_course_document.side_effect = Exception("Processing failed")
        
        rag = RAGSystem(test_config)
        
        # Test the method
        course, chunk_count = rag.add_course_document("/path/to/bad_course.txt")
        
        assert course is None
        assert chunk_count == 0

    @patch('rag_system.os.path.exists')
    @patch('rag_system.os.path.isfile')
    @patch('rag_system.os.path.join')
    @patch('rag_system.os.listdir')
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator') 
    @patch('rag_system.SessionManager')
    def test_add_course_folder_success(self, mock_session_manager, mock_ai_generator,
                                     mock_vector_store, mock_document_processor,
                                     mock_listdir, mock_join, mock_isfile, mock_exists, test_config,
                                     sample_course, sample_course_chunks):
        """Test successful course folder addition"""
        # Setup mocks
        mock_exists.return_value = True
        mock_listdir.return_value = ["course1.txt", "course2.pdf", "README.md"]
        mock_isfile.return_value = True  # All files exist
        mock_join.side_effect = lambda a, b: f"{a}/{b}"  # Simple path join mock
        
        mock_doc_processor = Mock()
        mock_document_processor.return_value = mock_doc_processor
        
        # Create a second course for the second file
        sample_course2 = Course(
            title="Different Course Title",
            instructor="Different Instructor",
            course_link="https://example.com/course2"
        )
        sample_chunks2 = [
            CourseChunk(
                content="Different course content",
                course_title="Different Course Title",
                lesson_number=1,
                chunk_index=0
            )
        ]
        
        # Return different course data for each file
        mock_doc_processor.process_course_document.side_effect = [
            (sample_course, sample_course_chunks),      # First file (course1.txt)
            (sample_course2, sample_chunks2),           # Second file (course2.pdf)
        ]
        
        mock_store = Mock()
        mock_vector_store.return_value = mock_store
        mock_store.get_existing_course_titles.return_value = []  # No existing courses
        
        rag = RAGSystem(test_config)
        
        # Test the method
        total_courses, total_chunks = rag.add_course_folder("/path/to/courses")
        
        assert total_courses == 2  # txt and pdf files
        assert total_chunks == 4  # 3 chunks from first course + 1 chunk from second course

    @patch('rag_system.os.path.exists')
    def test_add_course_folder_nonexistent(self, mock_exists, test_config):
        """Test course folder addition with nonexistent folder"""
        mock_exists.return_value = False
        
        with patch('rag_system.VectorStore'), \
             patch('rag_system.DocumentProcessor'), \
             patch('rag_system.AIGenerator'), \
             patch('rag_system.SessionManager'):
            
            rag = RAGSystem(test_config)
            total_courses, total_chunks = rag.add_course_folder("/nonexistent/path")
            
            assert total_courses == 0
            assert total_chunks == 0

    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    def test_query_without_session(self, mock_session_manager, mock_ai_generator,
                                 mock_vector_store, mock_document_processor, test_config):
        """Test query without session ID"""
        # Setup mocks
        mock_ai = Mock()
        mock_ai_generator.return_value = mock_ai
        mock_ai.generate_response.return_value = "Test response"
        
        mock_session = Mock()
        mock_session_manager.return_value = mock_session
        mock_session.get_conversation_history.return_value = None
        
        rag = RAGSystem(test_config)
        rag.tool_manager = Mock()
        rag.tool_manager.get_tool_definitions.return_value = []
        rag.tool_manager.get_last_sources.return_value = []
        rag.tool_manager.reset_sources = Mock()
        
        # Test the method
        response, sources = rag.query("What is machine learning?")
        
        assert response == "Test response"
        assert sources == []
        
        # Check that AI generator was called correctly
        mock_ai.generate_response.assert_called_once()
        call_args = mock_ai.generate_response.call_args
        assert "What is machine learning?" in call_args[1]["query"]
        assert call_args[1]["conversation_history"] is None

    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    def test_query_with_session(self, mock_session_manager, mock_ai_generator,
                              mock_vector_store, mock_document_processor, test_config):
        """Test query with session ID"""
        # Setup mocks
        mock_ai = Mock()
        mock_ai_generator.return_value = mock_ai
        mock_ai.generate_response.return_value = "Response with context"
        
        mock_session = Mock()
        mock_session_manager.return_value = mock_session
        mock_session.get_conversation_history.return_value = "Previous context"
        
        rag = RAGSystem(test_config)
        rag.tool_manager = Mock()
        rag.tool_manager.get_tool_definitions.return_value = []
        rag.tool_manager.get_last_sources.return_value = [{"text": "Source 1", "link": None}]
        rag.tool_manager.reset_sources = Mock()
        
        # Test the method  
        response, sources = rag.query("Follow-up question", session_id="session_123")
        
        assert response == "Response with context"
        assert len(sources) == 1
        assert sources[0]["text"] == "Source 1"
        
        # Check that session methods were called
        mock_session.get_conversation_history.assert_called_once_with("session_123")
        mock_session.add_exchange.assert_called_once_with("session_123", "Follow-up question", "Response with context")

    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    def test_query_with_sources(self, mock_session_manager, mock_ai_generator,
                              mock_vector_store, mock_document_processor, test_config):
        """Test query that returns sources"""
        # Setup mocks
        mock_ai = Mock()
        mock_ai_generator.return_value = mock_ai
        mock_ai.generate_response.return_value = "Response based on sources"
        
        rag = RAGSystem(test_config)
        rag.tool_manager = Mock()
        rag.tool_manager.get_tool_definitions.return_value = [{"name": "search_course_content"}]
        rag.tool_manager.get_last_sources.return_value = [
            {"text": "Course A - Lesson 1", "link": "https://example.com/lesson1"},
            {"text": "Course B", "link": None}
        ]
        rag.tool_manager.reset_sources = Mock()
        
        # Test the method
        response, sources = rag.query("Tell me about AI")
        
        assert response == "Response based on sources"
        assert len(sources) == 2
        assert sources[0]["text"] == "Course A - Lesson 1"
        assert sources[0]["link"] == "https://example.com/lesson1"
        assert sources[1]["text"] == "Course B"
        assert sources[1]["link"] is None
        
        # Check that sources were reset
        rag.tool_manager.reset_sources.assert_called_once()

    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    def test_get_course_analytics(self, mock_session_manager, mock_ai_generator,
                                mock_vector_store, mock_document_processor, test_config):
        """Test course analytics retrieval"""
        # Setup mocks
        mock_store = Mock()
        mock_vector_store.return_value = mock_store
        mock_store.get_course_count.return_value = 5
        mock_store.get_existing_course_titles.return_value = [
            "Course A", "Course B", "Course C", "Course D", "Course E"
        ]
        
        rag = RAGSystem(test_config)
        
        # Test the method
        analytics = rag.get_course_analytics()
        
        assert analytics["total_courses"] == 5
        assert len(analytics["course_titles"]) == 5
        assert "Course A" in analytics["course_titles"]

    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    def test_tool_registration(self, mock_session_manager, mock_ai_generator,
                             mock_vector_store, mock_document_processor, test_config):
        """Test that tools are properly registered"""
        rag = RAGSystem(test_config)
        
        # Check that search and outline tools are registered
        assert "search_course_content" in rag.tool_manager.tools
        assert "get_course_outline" in rag.tool_manager.tools
        
        # Check that tool definitions are available
        definitions = rag.tool_manager.get_tool_definitions()
        tool_names = [def_["name"] for def_ in definitions]
        assert "search_course_content" in tool_names
        assert "get_course_outline" in tool_names

    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    def test_query_error_handling(self, mock_session_manager, mock_ai_generator,
                                mock_vector_store, mock_document_processor, test_config):
        """Test query error handling"""
        # Setup mocks to raise exception
        mock_ai = Mock()
        mock_ai_generator.return_value = mock_ai
        mock_ai.generate_response.side_effect = Exception("AI Generation failed")
        
        rag = RAGSystem(test_config)
        rag.tool_manager = Mock()
        
        # Test that exception is propagated
        with pytest.raises(Exception) as exc_info:
            rag.query("Test query")
        
        assert "AI Generation failed" in str(exc_info.value)

    def test_prompt_formatting(self, test_config):
        """Test that query prompts are properly formatted"""
        with patch('rag_system.VectorStore'), \
             patch('rag_system.DocumentProcessor'), \
             patch('rag_system.AIGenerator') as mock_ai_generator, \
             patch('rag_system.SessionManager'):
            
            mock_ai = Mock()
            mock_ai_generator.return_value = mock_ai
            mock_ai.generate_response.return_value = "Response"
            
            rag = RAGSystem(test_config)
            rag.tool_manager = Mock()
            rag.tool_manager.get_tool_definitions.return_value = []
            rag.tool_manager.get_last_sources.return_value = []
            rag.tool_manager.reset_sources = Mock()
            
            rag.query("What is machine learning?")
            
            # Check that prompt includes the expected format
            call_args = mock_ai.generate_response.call_args
            query_arg = call_args[1]["query"]
            assert "Answer this question about course materials:" in query_arg
            assert "What is machine learning?" in query_arg