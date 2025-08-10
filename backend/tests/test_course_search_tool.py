"""Tests for CourseSearchTool functionality"""
import pytest
from unittest.mock import Mock, patch
from search_tools import CourseSearchTool, ToolManager
from vector_store import SearchResults


class TestCourseSearchTool:
    """Test cases for CourseSearchTool"""

    def test_get_tool_definition(self):
        """Test that tool definition is correctly structured"""
        mock_vector_store = Mock()
        tool = CourseSearchTool(mock_vector_store)
        
        definition = tool.get_tool_definition()
        
        assert definition["name"] == "search_course_content"
        assert "description" in definition
        assert "input_schema" in definition
        assert definition["input_schema"]["required"] == ["query"]
        assert "query" in definition["input_schema"]["properties"]
        assert "course_name" in definition["input_schema"]["properties"]
        assert "lesson_number" in definition["input_schema"]["properties"]

    def test_execute_with_successful_search(self, mock_vector_store, sample_search_results):
        """Test successful search execution"""
        mock_vector_store.search.return_value = sample_search_results
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson/0"
        
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("test query")
        
        assert result is not None
        assert len(result) > 0
        assert "Building Towards Computer Use with Anthropic" in result
        assert "[Building Towards Computer Use with Anthropic - Lesson 0]" in result
        
        # Check that search was called correctly
        mock_vector_store.search.assert_called_once_with(
            query="test query",
            course_name=None,
            lesson_number=None
        )

    def test_execute_with_course_name_filter(self, mock_vector_store, sample_search_results):
        """Test search with course name filter"""
        mock_vector_store.search.return_value = sample_search_results
        
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("test query", course_name="Anthropic Course")
        
        mock_vector_store.search.assert_called_once_with(
            query="test query",
            course_name="Anthropic Course",
            lesson_number=None
        )

    def test_execute_with_lesson_number_filter(self, mock_vector_store, sample_search_results):
        """Test search with lesson number filter"""
        mock_vector_store.search.return_value = sample_search_results
        
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("test query", lesson_number=1)
        
        mock_vector_store.search.assert_called_once_with(
            query="test query",
            course_name=None,
            lesson_number=1
        )

    def test_execute_with_both_filters(self, mock_vector_store, sample_search_results):
        """Test search with both course name and lesson number filters"""
        mock_vector_store.search.return_value = sample_search_results
        
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("test query", course_name="Test Course", lesson_number=2)
        
        mock_vector_store.search.assert_called_once_with(
            query="test query",
            course_name="Test Course",
            lesson_number=2
        )

    def test_execute_with_empty_results(self, mock_vector_store, sample_empty_results):
        """Test handling of empty search results"""
        mock_vector_store.search.return_value = sample_empty_results
        
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("nonexistent query")
        
        assert "No relevant content found" in result

    def test_execute_with_empty_results_and_filters(self, mock_vector_store, sample_empty_results):
        """Test empty results with filter information"""
        mock_vector_store.search.return_value = sample_empty_results
        
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("test query", course_name="Missing Course", lesson_number=99)
        
        assert "No relevant content found" in result
        assert "in course 'Missing Course'" in result
        assert "in lesson 99" in result

    def test_execute_with_search_error(self, mock_vector_store, sample_error_results):
        """Test handling of search errors"""
        mock_vector_store.search.return_value = sample_error_results
        
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("test query")
        
        assert result == "Search error occurred"

    def test_format_results_with_lesson_links(self, mock_vector_store):
        """Test result formatting with lesson links"""
        search_results = SearchResults(
            documents=["Sample content from lesson"],
            metadata=[{"course_title": "Test Course", "lesson_number": 1}],
            distances=[0.1]
        )
        mock_vector_store.search.return_value = search_results
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson/1"
        
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("test query")
        
        assert "[Test Course - Lesson 1]" in result
        assert "Sample content from lesson" in result
        
        # Check that sources are tracked
        assert len(tool.last_sources) == 1
        assert tool.last_sources[0]["text"] == "Test Course - Lesson 1"
        assert tool.last_sources[0]["link"] == "https://example.com/lesson/1"

    def test_format_results_without_lesson_links(self, mock_vector_store):
        """Test result formatting without lesson links"""
        search_results = SearchResults(
            documents=["Content without lesson number"],
            metadata=[{"course_title": "Test Course"}],
            distances=[0.1]
        )
        mock_vector_store.search.return_value = search_results
        mock_vector_store.get_lesson_link.return_value = None
        
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("test query")
        
        assert "[Test Course]" in result
        assert "Content without lesson number" in result
        
        # Check sources
        assert len(tool.last_sources) == 1
        assert tool.last_sources[0]["text"] == "Test Course"
        assert tool.last_sources[0]["link"] is None

    def test_sources_reset_between_searches(self, mock_vector_store, sample_search_results):
        """Test that sources are properly reset between searches"""
        mock_vector_store.search.return_value = sample_search_results
        
        tool = CourseSearchTool(mock_vector_store)
        
        # First search
        tool.execute("first query")
        first_sources = tool.last_sources.copy()
        
        # Second search
        tool.execute("second query")
        second_sources = tool.last_sources
        
        assert len(first_sources) > 0
        assert len(second_sources) > 0
        # Sources should be overwritten, not accumulated
        assert len(second_sources) == len(sample_search_results.documents)


class TestToolManager:
    """Test cases for ToolManager"""

    def test_register_tool(self, mock_vector_store):
        """Test tool registration"""
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        
        manager.register_tool(tool)
        
        assert "search_course_content" in manager.tools
        assert manager.tools["search_course_content"] == tool

    def test_get_tool_definitions(self, mock_vector_store):
        """Test getting all tool definitions"""
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)
        
        definitions = manager.get_tool_definitions()
        
        assert len(definitions) == 1
        assert definitions[0]["name"] == "search_course_content"

    def test_execute_tool_success(self, mock_vector_store, sample_search_results):
        """Test successful tool execution"""
        mock_vector_store.search.return_value = sample_search_results
        
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)
        
        result = manager.execute_tool("search_course_content", query="test query")
        
        assert result is not None
        assert len(result) > 0

    def test_execute_tool_not_found(self, mock_vector_store):
        """Test execution of non-existent tool"""
        manager = ToolManager()
        
        result = manager.execute_tool("nonexistent_tool", query="test")
        
        assert "Tool 'nonexistent_tool' not found" in result

    def test_get_last_sources(self, mock_vector_store, sample_search_results):
        """Test getting last sources from tools"""
        mock_vector_store.search.return_value = sample_search_results
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson/0"
        
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)
        
        # Execute a search
        manager.execute_tool("search_course_content", query="test query")
        
        sources = manager.get_last_sources()
        assert len(sources) > 0
        assert all("text" in source for source in sources)

    def test_reset_sources(self, mock_vector_store, sample_search_results):
        """Test resetting sources"""
        mock_vector_store.search.return_value = sample_search_results
        
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)
        
        # Execute a search
        manager.execute_tool("search_course_content", query="test query")
        assert len(manager.get_last_sources()) > 0
        
        # Reset sources
        manager.reset_sources()
        assert len(manager.get_last_sources()) == 0