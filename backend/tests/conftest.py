"""Shared pytest fixtures for RAG system tests"""

import os
import shutil

# Import system modules
import sys
import tempfile
from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock

import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from config import Config
from models import Course, CourseChunk, Lesson
from vector_store import SearchResults, VectorStore


@pytest.fixture
def test_config():
    """Test configuration with proper MAX_RESULTS"""
    config = Config()
    config.MAX_RESULTS = 5  # Fix the zero value
    config.CHROMA_PATH = "./test_chroma_db"
    config.ANTHROPIC_API_KEY = "test-key"
    return config


@pytest.fixture
def sample_course():
    """Sample course data for testing"""
    return Course(
        title="Building Towards Computer Use with Anthropic",
        course_link="https://www.deeplearning.ai/short-courses/building-toward-computer-use-with-anthropic/",
        instructor="Colt Steele",
        lessons=[
            Lesson(
                lesson_number=0,
                title="Introduction",
                lesson_link="https://learn.deeplearning.ai/courses/building-toward-computer-use-with-anthropic/lesson/a6k0z/introduction",
            ),
            Lesson(
                lesson_number=1,
                title="Getting Started",
                lesson_link="https://learn.deeplearning.ai/courses/building-toward-computer-use-with-anthropic/lesson/1/getting-started",
            ),
        ],
    )


@pytest.fixture
def sample_course_chunks(sample_course):
    """Sample course chunks for testing"""
    return [
        CourseChunk(
            content="Welcome to Building Toward Computer Use with Anthropic. Built in partnership with Anthropic and taught by Colt Steele.",
            course_title=sample_course.title,
            lesson_number=0,
            chunk_index=0,
        ),
        CourseChunk(
            content="In this course, you will learn how to use many of the models and features that all combine to enable computer use.",
            course_title=sample_course.title,
            lesson_number=0,
            chunk_index=1,
        ),
        CourseChunk(
            content="Getting started with Anthropic API is simple. First, you need an API key.",
            course_title=sample_course.title,
            lesson_number=1,
            chunk_index=2,
        ),
    ]


@pytest.fixture
def sample_search_results():
    """Sample search results for testing"""
    return SearchResults(
        documents=[
            "Welcome to Building Toward Computer Use with Anthropic.",
            "In this course, you will learn about computer use features.",
        ],
        metadata=[
            {
                "course_title": "Building Towards Computer Use with Anthropic",
                "lesson_number": 0,
            },
            {
                "course_title": "Building Towards Computer Use with Anthropic",
                "lesson_number": 0,
            },
        ],
        distances=[0.1, 0.2],
    )


@pytest.fixture
def sample_empty_results():
    """Sample empty search results"""
    return SearchResults(documents=[], metadata=[], distances=[])


@pytest.fixture
def sample_error_results():
    """Sample error search results"""
    return SearchResults.empty("Search error occurred")


@pytest.fixture
def mock_vector_store():
    """Mock vector store for testing"""
    mock_store = Mock(spec=VectorStore)
    mock_store.search.return_value = SearchResults(
        documents=["Sample document content"],
        metadata=[{"course_title": "Test Course", "lesson_number": 1}],
        distances=[0.1],
    )
    mock_store.get_lesson_link.return_value = "https://example.com/lesson/1"
    return mock_store


@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic client for testing"""
    mock_client = Mock()

    # Mock successful response without tools
    mock_response = Mock()
    mock_response.content = [Mock(text="This is a test response")]
    mock_response.stop_reason = "end_turn"
    mock_client.messages.create.return_value = mock_response

    return mock_client


@pytest.fixture
def mock_tool_use_response():
    """Mock Anthropic response with tool use"""
    mock_response = Mock()
    mock_response.stop_reason = "tool_use"

    # Mock tool use content block
    mock_tool_block = Mock()
    mock_tool_block.type = "tool_use"
    mock_tool_block.name = "search_course_content"
    mock_tool_block.id = "tool_123"
    mock_tool_block.input = {"query": "test query"}

    mock_response.content = [mock_tool_block]
    return mock_response


@pytest.fixture
def mock_final_response():
    """Mock final response after tool use"""
    mock_response = Mock()
    mock_response.content = [
        Mock(text="Here is the answer based on the search results")
    ]
    return mock_response


@pytest.fixture
def temp_chroma_dir():
    """Temporary directory for ChromaDB testing"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def test_course_document():
    """Sample course document content for testing"""
    return """Course Title: Test Course
Course Link: https://example.com/course
Course Instructor: Test Instructor

Lesson 1: Introduction
Lesson Link: https://example.com/lesson1
This is the introduction lesson content. It covers basic concepts.

Lesson 2: Advanced Topics
Lesson Link: https://example.com/lesson2
This lesson covers more advanced topics and practical examples.
"""


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Setup test environment"""
    # Ensure we're in the correct directory
    os.chdir("/Users/keng/repos/starting-ragchatbot-codebase/backend")
    yield
    # Cleanup if needed


@pytest.fixture
def mock_session_manager():
    """Mock session manager for testing"""
    mock_manager = Mock()
    mock_manager.create_session.return_value = "test_session_123"
    mock_manager.get_conversation_history.return_value = "Previous conversation history"
    return mock_manager


@pytest.fixture
def sample_tool_definitions():
    """Sample tool definitions for testing"""
    return [
        {
            "name": "search_course_content",
            "description": "Search course materials",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search for"}
                },
                "required": ["query"],
            },
        }
    ]
