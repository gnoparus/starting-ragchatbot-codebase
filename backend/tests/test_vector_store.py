"""Tests for VectorStore functionality"""

import os
import shutil
import tempfile
from unittest.mock import MagicMock, Mock, patch

import pytest
from models import Course, CourseChunk, Lesson
from vector_store import SearchResults, VectorStore


class TestSearchResults:
    """Test cases for SearchResults class"""

    def test_from_chroma_with_results(self):
        """Test creating SearchResults from ChromaDB results"""
        chroma_results = {
            "documents": [["doc1", "doc2"]],
            "metadatas": [[{"course": "A"}, {"course": "B"}]],
            "distances": [[0.1, 0.2]],
        }

        results = SearchResults.from_chroma(chroma_results)

        assert results.documents == ["doc1", "doc2"]
        assert results.metadata == [{"course": "A"}, {"course": "B"}]
        assert results.distances == [0.1, 0.2]
        assert results.error is None

    def test_from_chroma_empty(self):
        """Test creating SearchResults from empty ChromaDB results"""
        chroma_results = {"documents": [[]], "metadatas": [[]], "distances": [[]]}

        results = SearchResults.from_chroma(chroma_results)

        assert results.documents == []
        assert results.metadata == []
        assert results.distances == []

    def test_empty_with_error(self):
        """Test creating empty SearchResults with error message"""
        error_msg = "No results found"
        results = SearchResults.empty(error_msg)

        assert results.documents == []
        assert results.metadata == []
        assert results.distances == []
        assert results.error == error_msg

    def test_is_empty_true(self):
        """Test is_empty returns True for empty results"""
        results = SearchResults(documents=[], metadata=[], distances=[])
        assert results.is_empty() is True

    def test_is_empty_false(self):
        """Test is_empty returns False for non-empty results"""
        results = SearchResults(
            documents=["doc1"], metadata=[{"course": "A"}], distances=[0.1]
        )
        assert results.is_empty() is False


class TestVectorStore:
    """Test cases for VectorStore"""

    @patch("vector_store.chromadb.PersistentClient")
    @patch(
        "vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction"
    )
    def test_initialization(self, mock_embedding_fn, mock_client, temp_chroma_dir):
        """Test VectorStore initialization"""
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance

        mock_collection = Mock()
        mock_client_instance.get_or_create_collection.return_value = mock_collection

        store = VectorStore(temp_chroma_dir, "test-model", max_results=10)

        assert store.max_results == 10
        mock_client.assert_called_once_with(
            path=temp_chroma_dir, settings=mock_client.call_args[1]["settings"]
        )

        # Should create two collections
        assert mock_client_instance.get_or_create_collection.call_count == 2

    @patch("vector_store.chromadb.PersistentClient")
    @patch(
        "vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction"
    )
    def test_search_without_filters(
        self, mock_embedding_fn, mock_client, temp_chroma_dir
    ):
        """Test search without course name or lesson filters"""
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance

        mock_content_collection = Mock()
        mock_client_instance.get_or_create_collection.side_effect = [
            Mock(),
            mock_content_collection,
        ]

        # Mock successful search
        mock_content_collection.query.return_value = {
            "documents": [["Found document content"]],
            "metadatas": [[{"course_title": "Test Course", "lesson_number": 1}]],
            "distances": [[0.1]],
        }

        store = VectorStore(temp_chroma_dir, "test-model")
        results = store.search("test query")

        assert not results.is_empty()
        assert results.documents == ["Found document content"]
        assert results.metadata == [{"course_title": "Test Course", "lesson_number": 1}]

        # Check that query was called correctly
        mock_content_collection.query.assert_called_once_with(
            query_texts=["test query"], n_results=5, where=None  # default max_results
        )

    @patch("vector_store.chromadb.PersistentClient")
    @patch(
        "vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction"
    )
    def test_search_with_course_filter(
        self, mock_embedding_fn, mock_client, temp_chroma_dir
    ):
        """Test search with course name filter"""
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance

        mock_catalog_collection = Mock()
        mock_content_collection = Mock()
        mock_client_instance.get_or_create_collection.side_effect = [
            mock_catalog_collection,
            mock_content_collection,
        ]

        # Mock course resolution
        mock_catalog_collection.query.return_value = {
            "documents": [["Test Course"]],
            "metadatas": [[{"title": "Test Course"}]],
            "distances": [[0.1]],
        }

        # Mock content search
        mock_content_collection.query.return_value = {
            "documents": [["Course content"]],
            "metadatas": [[{"course_title": "Test Course"}]],
            "distances": [[0.1]],
        }

        store = VectorStore(temp_chroma_dir, "test-model")
        results = store.search("test query", course_name="Test")

        assert not results.is_empty()

        # Check that course resolution was called
        mock_catalog_collection.query.assert_called_once_with(
            query_texts=["Test"], n_results=1
        )

        # Check that content search used resolved course name
        mock_content_collection.query.assert_called_once_with(
            query_texts=["test query"],
            n_results=5,
            where={"course_title": "Test Course"},
        )

    @patch("vector_store.chromadb.PersistentClient")
    @patch(
        "vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction"
    )
    def test_search_with_lesson_filter(
        self, mock_embedding_fn, mock_client, temp_chroma_dir
    ):
        """Test search with lesson number filter"""
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance

        mock_content_collection = Mock()
        mock_client_instance.get_or_create_collection.side_effect = [
            Mock(),
            mock_content_collection,
        ]

        mock_content_collection.query.return_value = {
            "documents": [["Lesson content"]],
            "metadatas": [[{"course_title": "Test Course", "lesson_number": 2}]],
            "distances": [[0.1]],
        }

        store = VectorStore(temp_chroma_dir, "test-model")
        results = store.search("test query", lesson_number=2)

        assert not results.is_empty()

        mock_content_collection.query.assert_called_once_with(
            query_texts=["test query"], n_results=5, where={"lesson_number": 2}
        )

    @patch("vector_store.chromadb.PersistentClient")
    @patch(
        "vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction"
    )
    def test_search_with_both_filters(
        self, mock_embedding_fn, mock_client, temp_chroma_dir
    ):
        """Test search with both course name and lesson number filters"""
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance

        mock_catalog_collection = Mock()
        mock_content_collection = Mock()
        mock_client_instance.get_or_create_collection.side_effect = [
            mock_catalog_collection,
            mock_content_collection,
        ]

        # Mock course resolution
        mock_catalog_collection.query.return_value = {
            "documents": [["Resolved Course"]],
            "metadatas": [[{"title": "Resolved Course"}]],
            "distances": [[0.1]],
        }

        mock_content_collection.query.return_value = {
            "documents": [["Specific content"]],
            "metadatas": [[{"course_title": "Resolved Course", "lesson_number": 3}]],
            "distances": [[0.1]],
        }

        store = VectorStore(temp_chroma_dir, "test-model")
        results = store.search("test query", course_name="Course", lesson_number=3)

        assert not results.is_empty()

        # Check that AND filter was used
        expected_filter = {
            "$and": [{"course_title": "Resolved Course"}, {"lesson_number": 3}]
        }
        mock_content_collection.query.assert_called_once_with(
            query_texts=["test query"], n_results=5, where=expected_filter
        )

    @patch("vector_store.chromadb.PersistentClient")
    @patch(
        "vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction"
    )
    def test_search_course_not_found(
        self, mock_embedding_fn, mock_client, temp_chroma_dir
    ):
        """Test search when course name cannot be resolved"""
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance

        mock_catalog_collection = Mock()
        mock_client_instance.get_or_create_collection.side_effect = [
            mock_catalog_collection,
            Mock(),
        ]

        # Mock empty course resolution result
        mock_catalog_collection.query.return_value = {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

        store = VectorStore(temp_chroma_dir, "test-model")
        results = store.search("test query", course_name="Nonexistent Course")

        assert results.error == "No course found matching 'Nonexistent Course'"

    @patch("vector_store.chromadb.PersistentClient")
    @patch(
        "vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction"
    )
    def test_search_error_handling(
        self, mock_embedding_fn, mock_client, temp_chroma_dir
    ):
        """Test search error handling"""
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance

        mock_content_collection = Mock()
        mock_client_instance.get_or_create_collection.side_effect = [
            Mock(),
            mock_content_collection,
        ]

        # Mock search error
        mock_content_collection.query.side_effect = Exception("ChromaDB error")

        store = VectorStore(temp_chroma_dir, "test-model")
        results = store.search("test query")

        assert results.error == "Search error: ChromaDB error"

    @patch("vector_store.chromadb.PersistentClient")
    @patch(
        "vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction"
    )
    def test_add_course_metadata(
        self, mock_embedding_fn, mock_client, temp_chroma_dir, sample_course
    ):
        """Test adding course metadata to vector store"""
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance

        mock_catalog_collection = Mock()
        mock_client_instance.get_or_create_collection.side_effect = [
            mock_catalog_collection,
            Mock(),
        ]

        store = VectorStore(temp_chroma_dir, "test-model")
        store.add_course_metadata(sample_course)

        # Check that course was added to catalog
        mock_catalog_collection.add.assert_called_once()
        call_args = mock_catalog_collection.add.call_args

        assert call_args[1]["documents"] == [sample_course.title]
        assert call_args[1]["ids"] == [sample_course.title]

        metadata = call_args[1]["metadatas"][0]
        assert metadata["title"] == sample_course.title
        assert metadata["instructor"] == sample_course.instructor
        assert metadata["course_link"] == sample_course.course_link
        assert metadata["lesson_count"] == len(sample_course.lessons)
        assert "lessons_json" in metadata

    @patch("vector_store.chromadb.PersistentClient")
    @patch(
        "vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction"
    )
    def test_add_course_content(
        self, mock_embedding_fn, mock_client, temp_chroma_dir, sample_course_chunks
    ):
        """Test adding course content chunks to vector store"""
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance

        mock_content_collection = Mock()
        mock_client_instance.get_or_create_collection.side_effect = [
            Mock(),
            mock_content_collection,
        ]

        store = VectorStore(temp_chroma_dir, "test-model")
        store.add_course_content(sample_course_chunks)

        # Check that chunks were added
        mock_content_collection.add.assert_called_once()
        call_args = mock_content_collection.add.call_args

        expected_documents = [chunk.content for chunk in sample_course_chunks]
        assert call_args[1]["documents"] == expected_documents

        expected_ids = [
            f"{chunk.course_title.replace(' ', '_')}_{chunk.chunk_index}"
            for chunk in sample_course_chunks
        ]
        assert call_args[1]["ids"] == expected_ids

    @patch("vector_store.chromadb.PersistentClient")
    @patch(
        "vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction"
    )
    def test_add_empty_course_content(
        self, mock_embedding_fn, mock_client, temp_chroma_dir
    ):
        """Test adding empty course content list"""
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance

        mock_content_collection = Mock()
        mock_client_instance.get_or_create_collection.side_effect = [
            Mock(),
            mock_content_collection,
        ]

        store = VectorStore(temp_chroma_dir, "test-model")
        store.add_course_content([])  # Empty list

        # Should not call add method
        mock_content_collection.add.assert_not_called()

    @patch("vector_store.chromadb.PersistentClient")
    @patch(
        "vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction"
    )
    def test_clear_all_data(self, mock_embedding_fn, mock_client, temp_chroma_dir):
        """Test clearing all data from vector store"""
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance

        store = VectorStore(temp_chroma_dir, "test-model")
        store.clear_all_data()

        # Check that both collections were deleted
        expected_calls = [
            Mock(name="delete_collection")("course_catalog"),
            Mock(name="delete_collection")("course_content"),
        ]
        assert mock_client_instance.delete_collection.call_count == 2

    @patch("vector_store.chromadb.PersistentClient")
    @patch(
        "vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction"
    )
    def test_get_existing_course_titles(
        self, mock_embedding_fn, mock_client, temp_chroma_dir
    ):
        """Test getting existing course titles"""
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance

        mock_catalog_collection = Mock()
        mock_client_instance.get_or_create_collection.side_effect = [
            mock_catalog_collection,
            Mock(),
        ]

        mock_catalog_collection.get.return_value = {
            "ids": ["Course A", "Course B", "Course C"]
        }

        store = VectorStore(temp_chroma_dir, "test-model")
        titles = store.get_existing_course_titles()

        assert titles == ["Course A", "Course B", "Course C"]

    @patch("vector_store.chromadb.PersistentClient")
    @patch(
        "vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction"
    )
    def test_get_course_count(self, mock_embedding_fn, mock_client, temp_chroma_dir):
        """Test getting course count"""
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance

        mock_catalog_collection = Mock()
        mock_client_instance.get_or_create_collection.side_effect = [
            mock_catalog_collection,
            Mock(),
        ]

        mock_catalog_collection.get.return_value = {"ids": ["Course A", "Course B"]}

        store = VectorStore(temp_chroma_dir, "test-model")
        count = store.get_course_count()

        assert count == 2

    def test_build_filter_no_filters(self, temp_chroma_dir):
        """Test filter building with no filters"""
        with (
            patch("vector_store.chromadb.PersistentClient"),
            patch(
                "vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction"
            ),
        ):

            store = VectorStore(temp_chroma_dir, "test-model")
            filter_dict = store._build_filter(None, None)

            assert filter_dict is None

    def test_build_filter_course_only(self, temp_chroma_dir):
        """Test filter building with course name only"""
        with (
            patch("vector_store.chromadb.PersistentClient"),
            patch(
                "vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction"
            ),
        ):

            store = VectorStore(temp_chroma_dir, "test-model")
            filter_dict = store._build_filter("Test Course", None)

            assert filter_dict == {"course_title": "Test Course"}

    def test_build_filter_lesson_only(self, temp_chroma_dir):
        """Test filter building with lesson number only"""
        with (
            patch("vector_store.chromadb.PersistentClient"),
            patch(
                "vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction"
            ),
        ):

            store = VectorStore(temp_chroma_dir, "test-model")
            filter_dict = store._build_filter(None, 5)

            assert filter_dict == {"lesson_number": 5}

    def test_build_filter_both(self, temp_chroma_dir):
        """Test filter building with both course and lesson"""
        with (
            patch("vector_store.chromadb.PersistentClient"),
            patch(
                "vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction"
            ),
        ):

            store = VectorStore(temp_chroma_dir, "test-model")
            filter_dict = store._build_filter("Test Course", 3)

            expected = {"$and": [{"course_title": "Test Course"}, {"lesson_number": 3}]}
            assert filter_dict == expected
