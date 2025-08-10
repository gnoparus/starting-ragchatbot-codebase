"""Integration tests for the complete RAG system"""
import pytest
import tempfile
import shutil
import json
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient
from app import app, rag_system
from models import Course, Lesson, CourseChunk


class TestRAGIntegration:
    """Integration tests for the complete RAG system flow"""

    @patch('app.rag_system')
    def test_api_query_endpoint_success(self, mock_rag_system):
        """Test successful API query through the complete flow"""
        # Setup mock RAG system
        mock_rag_system.session_manager.create_session.return_value = "test_session_123"
        mock_rag_system.query.return_value = (
            "Here is the answer about machine learning",
            [
                {"text": "ML Course - Lesson 1", "link": "https://example.com/lesson1"},
                {"text": "AI Fundamentals", "link": None}
            ]
        )
        
        client = TestClient(app)
        
        response = client.post("/api/query", json={
            "query": "What is machine learning?"
        })
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["answer"] == "Here is the answer about machine learning"
        assert len(data["sources"]) == 2
        assert data["sources"][0]["text"] == "ML Course - Lesson 1"
        assert data["sources"][0]["link"] == "https://example.com/lesson1"
        assert data["session_id"] == "test_session_123"

    @patch('app.rag_system')
    def test_api_query_with_session(self, mock_rag_system):
        """Test API query with existing session"""
        mock_rag_system.query.return_value = (
            "Follow-up answer with context",
            []
        )
        
        client = TestClient(app)
        
        response = client.post("/api/query", json={
            "query": "Tell me more about that",
            "session_id": "existing_session_456"
        })
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["answer"] == "Follow-up answer with context"
        assert data["session_id"] == "existing_session_456"
        
        # Check that query was called with the session
        mock_rag_system.query.assert_called_once_with(
            "Tell me more about that", 
            "existing_session_456"
        )

    @patch('app.rag_system')
    def test_api_query_error_handling(self, mock_rag_system):
        """Test API error handling"""
        mock_rag_system.query.side_effect = Exception("Internal RAG error")
        
        client = TestClient(app)
        
        response = client.post("/api/query", json={
            "query": "This will fail"
        })
        
        assert response.status_code == 500
        assert "Internal RAG error" in response.json()["detail"]

    @patch('app.rag_system')
    def test_api_courses_endpoint(self, mock_rag_system):
        """Test courses statistics endpoint"""
        mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 3,
            "course_titles": ["Course A", "Course B", "Course C"]
        }
        
        client = TestClient(app)
        
        response = client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["total_courses"] == 3
        assert len(data["course_titles"]) == 3
        assert "Course A" in data["course_titles"]

    def test_complete_rag_flow_with_real_components(self, test_config, temp_chroma_dir, test_course_document):
        """Test complete RAG flow with real components (but mocked Claude API)"""
        # Create a temporary course file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(test_course_document)
            course_file = f.name
        
        try:
            # Update config for testing
            test_config.CHROMA_PATH = temp_chroma_dir
            test_config.MAX_RESULTS = 3  # Fix the zero issue
            
            # Mock the Claude API
            with patch('ai_generator.anthropic.Anthropic') as mock_anthropic:
                # Mock successful tool use response
                mock_client = Mock()
                mock_anthropic.return_value = mock_client
                
                # First response - tool use
                mock_tool_response = Mock()
                mock_tool_response.stop_reason = "tool_use"
                mock_tool_block = Mock()
                mock_tool_block.type = "tool_use"
                mock_tool_block.name = "search_course_content"
                mock_tool_block.id = "tool_123"
                mock_tool_block.input = {"query": "introduction"}
                mock_tool_response.content = [mock_tool_block]
                
                # Second response - final answer
                mock_final_response = Mock()
                mock_final_response.content = [Mock(text="Based on the course content, the introduction covers basic concepts.")]
                
                mock_client.messages.create.side_effect = [mock_tool_response, mock_final_response]
                
                # Import and create RAG system with test config
                from rag_system import RAGSystem
                rag = RAGSystem(test_config)
                
                # Add the test course
                course, chunk_count = rag.add_course_document(course_file)
                assert course is not None
                assert chunk_count > 0
                
                # Query the system
                response, sources = rag.query("Tell me about the introduction")
                
                assert response == "Based on the course content, the introduction covers basic concepts."
                assert isinstance(sources, list)
                
                # Check that the Claude API was called correctly
                assert mock_client.messages.create.call_count == 2
                
        finally:
            # Cleanup
            import os
            os.unlink(course_file)

    def test_search_tool_integration_with_vector_store(self, test_config, temp_chroma_dir, sample_course, sample_course_chunks):
        """Test CourseSearchTool integration with VectorStore"""
        test_config.CHROMA_PATH = temp_chroma_dir
        test_config.MAX_RESULTS = 2
        
        # Create real VectorStore and SearchTool
        from vector_store import VectorStore
        from search_tools import CourseSearchTool
        
        store = VectorStore(test_config.CHROMA_PATH, test_config.EMBEDDING_MODEL, test_config.MAX_RESULTS)
        tool = CourseSearchTool(store)
        
        # Add sample data
        store.add_course_metadata(sample_course)
        store.add_course_content(sample_course_chunks)
        
        # Test search functionality
        result = tool.execute("Anthropic computer use")
        
        assert result is not None
        assert len(result) > 0
        assert "Building Towards Computer Use with Anthropic" in result
        
        # Test with course filter
        result_filtered = tool.execute("getting started", course_name="Building")
        assert result_filtered is not None

    def test_ai_generator_tool_calling_integration(self, test_config):
        """Test AI Generator with real tool manager integration"""
        with patch('ai_generator.anthropic.Anthropic') as mock_anthropic:
            mock_client = Mock()
            mock_anthropic.return_value = mock_client
            
            # Mock tool use response
            mock_tool_response = Mock()
            mock_tool_response.stop_reason = "tool_use"
            mock_tool_block = Mock()
            mock_tool_block.type = "tool_use"
            mock_tool_block.name = "search_course_content"
            mock_tool_block.id = "tool_123"
            mock_tool_block.input = {"query": "test search"}
            mock_tool_response.content = [mock_tool_block]
            
            # Mock final response
            mock_final_response = Mock()
            mock_final_response.content = [Mock(text="Here's the answer based on search results")]
            
            mock_client.messages.create.side_effect = [mock_tool_response, mock_final_response]
            
            # Create real components
            from ai_generator import AIGenerator
            from search_tools import ToolManager, CourseSearchTool
            from vector_store import VectorStore
            
            # Setup
            ai_gen = AIGenerator(test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL)
            mock_store = Mock()
            mock_store.search.return_value.is_empty.return_value = False
            mock_store.search.return_value.error = None
            mock_store.search.return_value.documents = ["Sample content"]
            mock_store.search.return_value.metadata = [{"course_title": "Test Course"}]
            
            tool_manager = ToolManager()
            search_tool = CourseSearchTool(mock_store)
            tool_manager.register_tool(search_tool)
            
            # Test the integration
            response = ai_gen.generate_response(
                "Tell me about machine learning",
                tools=tool_manager.get_tool_definitions(),
                tool_manager=tool_manager
            )
            
            assert response == "Here's the answer based on search results"
            
            # Verify tool was executed
            mock_store.search.assert_called_once_with(
                query="test search",
                course_name=None,
                lesson_number=None
            )

    def test_session_management_integration(self, test_config):
        """Test session management integration"""
        from session_manager import SessionManager
        from rag_system import RAGSystem
        
        # Mock other components  
        with patch('rag_system.VectorStore'), \
             patch('rag_system.DocumentProcessor'), \
             patch('rag_system.AIGenerator') as mock_ai_gen:
            
            mock_ai = Mock()
            mock_ai_gen.return_value = mock_ai
            mock_ai.generate_response.return_value = "Response with session context"
            
            rag = RAGSystem(test_config)
            rag.tool_manager = Mock()
            rag.tool_manager.get_tool_definitions.return_value = []
            rag.tool_manager.get_last_sources.return_value = []
            rag.tool_manager.reset_sources = Mock()
            
            # Test session creation and usage
            session_id = rag.session_manager.create_session()
            assert session_id is not None
            
            # First query
            response1, _ = rag.query("First question", session_id=session_id)
            
            # Second query (should have context)
            response2, _ = rag.query("Follow up question", session_id=session_id)
            
            assert response1 == "Response with session context"
            assert response2 == "Response with session context"
            
            # Check that history was used in second call
            assert mock_ai.generate_response.call_count == 2
            second_call = mock_ai.generate_response.call_args_list[1]
            assert second_call[1]['conversation_history'] is not None

    def test_error_propagation_through_layers(self, test_config):
        """Test error propagation from components up through the API"""
        # Test vector store error propagation
        with patch('app.rag_system') as mock_rag:
            mock_rag.query.side_effect = Exception("Vector store connection failed")
            
            client = TestClient(app)
            response = client.post("/api/query", json={"query": "test"})
            
            assert response.status_code == 500
            assert "Vector store connection failed" in response.json()["detail"]

    def test_max_results_zero_issue_simulation(self, test_config, temp_chroma_dir):
        """Test the MAX_RESULTS=0 issue that causes empty search results"""
        test_config.CHROMA_PATH = temp_chroma_dir
        test_config.MAX_RESULTS = 0  # This is the problematic configuration
        
        from vector_store import VectorStore
        from search_tools import CourseSearchTool
        
        store = VectorStore(test_config.CHROMA_PATH, test_config.EMBEDDING_MODEL, test_config.MAX_RESULTS)
        tool = CourseSearchTool(store)
        
        # Add some sample content  
        from models import Course, Lesson, CourseChunk
        
        course = Course(title="Test Course", instructor="Test Instructor")
        chunks = [
            CourseChunk(
                content="This is test content about machine learning",
                course_title="Test Course",
                lesson_number=1,
                chunk_index=0
            )
        ]
        
        store.add_course_metadata(course)
        store.add_course_content(chunks)
        
        # This should return empty results due to MAX_RESULTS=0
        result = tool.execute("machine learning")
        
        # With MAX_RESULTS=0, we expect a ChromaDB error about zero results
        assert "Search error:" in result and ("cannot be negative, or zero" in result or "No relevant content found" in result)

    def test_max_results_fix_verification(self, test_config, temp_chroma_dir):
        """Test that fixing MAX_RESULTS resolves the issue"""
        test_config.CHROMA_PATH = temp_chroma_dir
        test_config.MAX_RESULTS = 5  # Fixed configuration
        
        from vector_store import VectorStore
        from search_tools import CourseSearchTool
        
        store = VectorStore(test_config.CHROMA_PATH, test_config.EMBEDDING_MODEL, test_config.MAX_RESULTS)
        tool = CourseSearchTool(store)
        
        # Add some sample content
        from models import Course, Lesson, CourseChunk
        
        course = Course(title="Test Course", instructor="Test Instructor")
        chunks = [
            CourseChunk(
                content="This is test content about machine learning algorithms and neural networks",
                course_title="Test Course",
                lesson_number=1,
                chunk_index=0
            )
        ]
        
        store.add_course_metadata(course)
        store.add_course_content(chunks)
        
        # This should now return actual results
        result = tool.execute("machine learning")
        
        # With proper MAX_RESULTS, we should get content
        assert "No relevant content found" not in result
        assert len(result) > 0
        assert "Test Course" in result

    @patch('app.rag_system')
    def test_api_request_validation(self, mock_rag_system):
        """Test API request validation"""
        client = TestClient(app)
        
        # Test missing query
        response = client.post("/api/query", json={})
        assert response.status_code == 422
        
        # Test invalid JSON
        response = client.post("/api/query", json={"query": 123})  # Should be string
        assert response.status_code == 422
        
        # Test valid request
        mock_rag_system.session_manager.create_session.return_value = "test_session"
        mock_rag_system.query.return_value = ("Answer", [])
        
        response = client.post("/api/query", json={"query": "Valid question"})
        assert response.status_code == 200