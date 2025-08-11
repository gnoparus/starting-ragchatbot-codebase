"""API endpoint tests for the RAG system FastAPI application"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import json

# Mark all tests in this file as API tests
pytestmark = pytest.mark.api


class TestQueryEndpoint:
    """Test the /api/query endpoint"""
    
    def test_query_with_session_id_success(self, client, sample_query_request, expected_query_response):
        """Test successful query with existing session ID"""
        response = client.post("/api/query", json=sample_query_request)
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        assert "answer" in data
        assert "sources" in data  
        assert "session_id" in data
        
        # Verify content matches expected
        assert data["answer"] == expected_query_response["answer"]
        assert data["session_id"] == expected_query_response["session_id"]
        assert len(data["sources"]) == 1
        assert data["sources"][0]["text"] == "Test source"
        assert data["sources"][0]["link"] == "https://example.com/test"
    
    def test_query_without_session_id_creates_new_session(self, client, sample_query_request_no_session):
        """Test query without session ID creates new session"""
        response = client.post("/api/query", json=sample_query_request_no_session)
        
        assert response.status_code == 200
        data = response.json()
        
        # Should create new session
        assert "session_id" in data
        assert data["session_id"] == "test_session_123"
        assert data["answer"] == "Test answer"
    
    def test_query_with_invalid_request_body(self, client):
        """Test query with invalid request body"""
        invalid_request = {"invalid_field": "value"}
        
        response = client.post("/api/query", json=invalid_request)
        
        assert response.status_code == 422  # Validation error
        
    def test_query_with_empty_query(self, client):
        """Test query with empty query string"""
        empty_query = {"query": ""}
        
        response = client.post("/api/query", json=empty_query)
        
        # Should still process (empty string is valid)
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
    
    def test_query_with_very_long_query(self, client):
        """Test query with very long query string"""
        long_query = {"query": "What is computer use? " * 100}  # Very long query
        
        response = client.post("/api/query", json=long_query)
        
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
    
    def test_query_with_special_characters(self, client):
        """Test query with special characters and unicode"""
        special_query = {
            "query": "What is 🤖 machine learning? Special chars: @#$%^&*()"
        }
        
        response = client.post("/api/query", json=special_query)
        
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
    
    def test_query_error_handling(self, client):
        """Test error handling when RAG system fails"""
        # Mock the RAG system to raise an error
        with patch.object(client.app.state.mock_rag, 'query', side_effect=Exception("RAG error")):
            response = client.post("/api/query", json={"query": "test"})
            
            assert response.status_code == 500
            data = response.json()
            assert "detail" in data
            assert "RAG error" in data["detail"]
    
    def test_query_response_headers(self, client, sample_query_request):
        """Test response headers are set correctly"""
        response = client.post("/api/query", json=sample_query_request)
        
        assert response.status_code == 200
        # Check basic headers are present
        assert "content-type" in response.headers
        assert response.headers["content-type"] == "application/json"
    
    def test_query_with_malformed_json(self, client):
        """Test query with malformed JSON"""
        response = client.post(
            "/api/query", 
            data="invalid json content",
            headers={"content-type": "application/json"}
        )
        
        assert response.status_code == 422


class TestCoursesEndpoint:
    """Test the /api/courses endpoint"""
    
    def test_get_course_stats_success(self, client, expected_course_stats):
        """Test successful retrieval of course statistics"""
        response = client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        assert "total_courses" in data
        assert "course_titles" in data
        
        # Verify content
        assert data["total_courses"] == expected_course_stats["total_courses"]
        assert data["course_titles"] == expected_course_stats["course_titles"]
        assert isinstance(data["course_titles"], list)
        assert len(data["course_titles"]) == 2
    
    def test_get_course_stats_error_handling(self, client):
        """Test error handling when analytics fails"""
        with patch.object(client.app.state.mock_rag, 'get_course_analytics', side_effect=Exception("Analytics error")):
            response = client.get("/api/courses")
            
            assert response.status_code == 500
            data = response.json()
            assert "detail" in data
            assert "Analytics error" in data["detail"]
    
    def test_get_course_stats_response_headers(self, client):
        """Test response headers for courses endpoint"""
        response = client.get("/api/courses")
        
        assert response.status_code == 200
        # Check basic headers
        assert "content-type" in response.headers
        assert response.headers["content-type"] == "application/json"
    
    def test_get_course_stats_no_params_required(self, client):
        """Test that courses endpoint requires no parameters"""
        response = client.get("/api/courses")
        
        assert response.status_code == 200
        # Should work without any query parameters


class TestRootEndpoint:
    """Test the root / endpoint"""
    
    def test_root_endpoint(self, client):
        """Test root endpoint returns test message"""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Test API"


class TestCORSAndMiddleware:
    """Test CORS middleware and other middleware functionality"""
    
    def test_cors_options_request(self, client):
        """Test OPTIONS request handling"""
        response = client.options("/api/query")
        
        # TestClient may return 405 for OPTIONS, which is expected behavior
        # The real server would handle CORS properly
        assert response.status_code in [200, 204, 405]
    
    def test_middleware_configuration(self, test_app):
        """Test that the app has middleware configured"""
        # Verify middleware is configured on the app
        assert hasattr(test_app, 'user_middleware')
        assert len(test_app.user_middleware) > 0  # Should have at least CORS middleware
    
    def test_app_accepts_json_requests(self, client):
        """Test that the app properly handles JSON requests"""
        response = client.post("/api/query", json={"query": "test"})
        
        assert response.status_code == 200
        # Should return JSON content type
        assert "application/json" in response.headers.get("content-type", "")
    
    def test_app_title_and_config(self, test_app):
        """Test that the test app is properly configured"""
        assert test_app.title == "Course Materials RAG System - Test"
        assert hasattr(test_app.state, "mock_rag")


class TestRequestValidation:
    """Test request validation and edge cases"""
    
    def test_post_query_with_extra_fields(self, client):
        """Test that extra fields in query request are ignored"""
        request_with_extra = {
            "query": "test query",
            "session_id": "test_session",
            "extra_field": "should be ignored"
        }
        
        response = client.post("/api/query", json=request_with_extra)
        
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
    
    def test_query_request_content_type(self, client):
        """Test different content types for query request"""
        # Should work with application/json
        response = client.post(
            "/api/query",
            json={"query": "test"},
            headers={"content-type": "application/json"}
        )
        assert response.status_code == 200
    
    def test_unsupported_http_methods(self, client):
        """Test unsupported HTTP methods on endpoints"""
        # GET on /api/query should not be allowed
        response = client.get("/api/query")
        assert response.status_code == 405  # Method Not Allowed
        
        # POST on /api/courses should not be allowed  
        response = client.post("/api/courses", json={"test": "data"})
        assert response.status_code == 405  # Method Not Allowed


class TestIntegrationScenarios:
    """Test integration scenarios and workflows"""
    
    def test_full_conversation_workflow(self, client):
        """Test a complete conversation workflow"""
        # First query without session
        response1 = client.post("/api/query", json={"query": "What is AI?"})
        assert response1.status_code == 200
        data1 = response1.json()
        session_id = data1["session_id"]
        
        # Second query with same session
        response2 = client.post("/api/query", json={
            "query": "Tell me more about that",
            "session_id": session_id
        })
        assert response2.status_code == 200
        data2 = response2.json()
        assert data2["session_id"] == session_id
    
    def test_multiple_concurrent_requests(self, client):
        """Test handling multiple concurrent requests"""
        import threading
        import queue
        
        results = queue.Queue()
        
        def make_request():
            response = client.post("/api/query", json={"query": "concurrent test"})
            results.put(response.status_code)
        
        # Make 5 concurrent requests
        threads = []
        for i in range(5):
            t = threading.Thread(target=make_request)
            threads.append(t)
            t.start()
        
        # Wait for all threads
        for t in threads:
            t.join()
        
        # Check all requests succeeded
        while not results.empty():
            status_code = results.get()
            assert status_code == 200
    
    def test_session_persistence_across_requests(self, client):
        """Test that session IDs are maintained across requests"""
        # Multiple requests with same session ID
        session_id = "persistent_session_123"
        
        for i in range(3):
            response = client.post("/api/query", json={
                "query": f"Question {i}",
                "session_id": session_id
            })
            assert response.status_code == 200
            data = response.json()
            assert data["session_id"] == session_id


class TestResponseStructure:
    """Test response structure and data types"""
    
    def test_query_response_schema_compliance(self, client):
        """Test that query response matches expected schema"""
        response = client.post("/api/query", json={"query": "test"})
        
        assert response.status_code == 200
        data = response.json()
        
        # Required fields
        required_fields = ["answer", "sources", "session_id"]
        for field in required_fields:
            assert field in data
        
        # Data types
        assert isinstance(data["answer"], str)
        assert isinstance(data["sources"], list)
        assert isinstance(data["session_id"], str)
        
        # Sources structure
        if data["sources"]:
            source = data["sources"][0]
            assert "text" in source
            assert isinstance(source["text"], str)
            # link is optional
            if "link" in source:
                assert isinstance(source["link"], (str, type(None)))
    
    def test_course_stats_response_schema_compliance(self, client):
        """Test that course stats response matches expected schema"""
        response = client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        
        # Required fields
        assert "total_courses" in data
        assert "course_titles" in data
        
        # Data types
        assert isinstance(data["total_courses"], int)
        assert isinstance(data["course_titles"], list)
        
        # All course titles should be strings
        for title in data["course_titles"]:
            assert isinstance(title, str)