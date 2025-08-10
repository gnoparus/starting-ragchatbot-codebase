"""Tests for AI Generator functionality"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from ai_generator import AIGenerator


class TestAIGenerator:
    """Test cases for AIGenerator"""

    def test_initialization(self):
        """Test AIGenerator initialization"""
        api_key = "test-api-key"
        model = "claude-sonnet-4-20250514"
        
        generator = AIGenerator(api_key, model)
        
        assert generator.model == model
        assert generator.base_params["model"] == model
        assert generator.base_params["temperature"] == 0
        assert generator.base_params["max_tokens"] == 800

    @patch('ai_generator.anthropic.Anthropic')
    def test_generate_response_without_tools(self, mock_anthropic_class, mock_anthropic_client):
        """Test response generation without tools"""
        mock_anthropic_class.return_value = mock_anthropic_client
        
        generator = AIGenerator("test-key", "test-model")
        query = "What is machine learning?"
        
        result = generator.generate_response(query)
        
        assert result == "This is a test response"
        mock_anthropic_client.messages.create.assert_called_once()
        
        # Check the call arguments
        call_args = mock_anthropic_client.messages.create.call_args
        assert call_args[1]["messages"][0]["content"] == query
        assert "tools" not in call_args[1]

    @patch('ai_generator.anthropic.Anthropic')
    def test_generate_response_with_conversation_history(self, mock_anthropic_class, mock_anthropic_client):
        """Test response generation with conversation history"""
        mock_anthropic_class.return_value = mock_anthropic_client
        
        generator = AIGenerator("test-key", "test-model")
        query = "What is AI?"
        history = "Previous conversation about technology"
        
        result = generator.generate_response(query, conversation_history=history)
        
        assert result == "This is a test response"
        
        # Check that history is included in system prompt
        call_args = mock_anthropic_client.messages.create.call_args
        system_content = call_args[1]["system"]
        assert "Previous conversation" in system_content

    @patch('ai_generator.anthropic.Anthropic')
    def test_generate_response_with_tools_no_tool_use(self, mock_anthropic_class, mock_anthropic_client, sample_tool_definitions):
        """Test response generation with tools but no tool use"""
        mock_anthropic_class.return_value = mock_anthropic_client
        
        generator = AIGenerator("test-key", "test-model")
        query = "Hello, how are you?"
        
        result = generator.generate_response(query, tools=sample_tool_definitions)
        
        assert result == "This is a test response"
        
        # Check that tools are included
        call_args = mock_anthropic_client.messages.create.call_args
        assert call_args[1]["tools"] == sample_tool_definitions
        assert call_args[1]["tool_choice"]["type"] == "auto"

    @patch('ai_generator.anthropic.Anthropic')
    def test_generate_response_with_tool_use(self, mock_anthropic_class, mock_tool_use_response, mock_final_response, sample_tool_definitions):
        """Test response generation with tool use"""
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        
        # First call returns tool use, second call returns final response
        mock_client.messages.create.side_effect = [mock_tool_use_response, mock_final_response]
        
        # Mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Search results: Found relevant content"
        
        generator = AIGenerator("test-key", "test-model")
        query = "Tell me about machine learning"
        
        result = generator.generate_response(
            query, 
            tools=sample_tool_definitions, 
            tool_manager=mock_tool_manager
        )
        
        assert result == "Here is the answer based on the search results"
        
        # Check that tool was executed
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content",
            query="test query"
        )
        
        # Check that two API calls were made
        assert mock_client.messages.create.call_count == 2

    def test_handle_tool_execution_single_tool(self, mock_tool_use_response, sample_tool_definitions):
        """Test tool execution handling with single tool"""
        # Mock the client for the final call
        with patch('ai_generator.anthropic.Anthropic') as mock_anthropic_class:
            mock_client = Mock()
            mock_anthropic_class.return_value = mock_client
            
            # Mock final response
            mock_final_response = Mock()
            mock_final_response.content = [Mock(text="Final answer after tool use")]
            mock_client.messages.create.return_value = mock_final_response
            
            generator = AIGenerator("test-key", "test-model")
            
            # Mock tool manager
            mock_tool_manager = Mock()
            mock_tool_manager.execute_tool.return_value = "Tool execution result"
            
            # Base parameters for the call
            base_params = {
                "model": "test-model",
                "messages": [{"role": "user", "content": "test query"}],
                "system": "test system prompt",
                "tools": sample_tool_definitions,
                "tool_choice": {"type": "auto"}
            }
            
            result = generator._handle_tool_execution(
                mock_tool_use_response, 
                base_params, 
                mock_tool_manager
            )
            
            assert result == "Final answer after tool use"
            mock_tool_manager.execute_tool.assert_called_once()

    def test_handle_tool_execution_multiple_tools(self, sample_tool_definitions):
        """Test tool execution handling with multiple tools"""
        # Create mock response with multiple tool uses
        mock_response = Mock()
        mock_response.stop_reason = "tool_use"
        
        # Multiple tool blocks
        tool_block_1 = Mock()
        tool_block_1.type = "tool_use"
        tool_block_1.name = "search_course_content"
        tool_block_1.id = "tool_1"
        tool_block_1.input = {"query": "first query"}
        
        tool_block_2 = Mock()
        tool_block_2.type = "tool_use"
        tool_block_2.name = "search_course_content"
        tool_block_2.id = "tool_2"
        tool_block_2.input = {"query": "second query"}
        
        mock_response.content = [tool_block_1, tool_block_2]
        
        with patch('ai_generator.anthropic.Anthropic') as mock_anthropic_class:
            mock_client = Mock()
            mock_anthropic_class.return_value = mock_client
            
            mock_final_response = Mock()
            mock_final_response.content = [Mock(text="Combined results")]
            mock_client.messages.create.return_value = mock_final_response
            
            generator = AIGenerator("test-key", "test-model")
            
            mock_tool_manager = Mock()
            mock_tool_manager.execute_tool.side_effect = ["Result 1", "Result 2"]
            
            base_params = {
                "model": "test-model",
                "messages": [{"role": "user", "content": "test query"}],
                "system": "test system prompt"
            }
            
            result = generator._handle_tool_execution(
                mock_response, 
                base_params, 
                mock_tool_manager
            )
            
            assert result == "Combined results"
            assert mock_tool_manager.execute_tool.call_count == 2

    @patch('ai_generator.anthropic.Anthropic')
    def test_generate_response_api_error(self, mock_anthropic_class):
        """Test handling of API errors"""
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        
        # Mock an API error
        mock_client.messages.create.side_effect = Exception("API Error")
        
        generator = AIGenerator("test-key", "test-model")
        
        with pytest.raises(Exception) as exc_info:
            generator.generate_response("test query")
        
        assert "API Error" in str(exc_info.value)

    def test_system_prompt_content(self):
        """Test that system prompt contains expected content"""
        generator = AIGenerator("test-key", "test-model")
        
        assert "course materials" in generator.SYSTEM_PROMPT.lower()
        assert "search tool" in generator.SYSTEM_PROMPT.lower()
        assert "outline tool" in generator.SYSTEM_PROMPT.lower()

    @patch('ai_generator.anthropic.Anthropic')
    def test_tool_manager_none_handling(self, mock_anthropic_class, mock_tool_use_response, sample_tool_definitions):
        """Test handling when tool_manager is None but tool use is requested"""
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        mock_client.messages.create.return_value = mock_tool_use_response
        
        generator = AIGenerator("test-key", "test-model")
        
        # Should now return an error message instead of crashing
        result = generator.generate_response(
            "test query", 
            tools=sample_tool_definitions, 
            tool_manager=None
        )
        
        assert result == "Error: Tool use requested but no tool manager provided"

    def test_base_params_configuration(self):
        """Test that base parameters are correctly configured"""
        generator = AIGenerator("test-key", "test-model-name")
        
        expected_params = {
            "model": "test-model-name",
            "temperature": 0,
            "max_tokens": 800
        }
        
        assert generator.base_params == expected_params

    @patch('ai_generator.anthropic.Anthropic')
    def test_message_structure_without_history(self, mock_anthropic_class, mock_anthropic_client):
        """Test message structure when no conversation history is provided"""
        mock_anthropic_class.return_value = mock_anthropic_client
        
        generator = AIGenerator("test-key", "test-model")
        generator.generate_response("test query")
        
        call_args = mock_anthropic_client.messages.create.call_args
        messages = call_args[1]["messages"]
        
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "test query"

    @patch('ai_generator.anthropic.Anthropic')  
    def test_system_message_without_history(self, mock_anthropic_class, mock_anthropic_client):
        """Test system message structure without conversation history"""
        mock_anthropic_class.return_value = mock_anthropic_client
        
        generator = AIGenerator("test-key", "test-model")
        generator.generate_response("test query")
        
        call_args = mock_anthropic_client.messages.create.call_args
        system_content = call_args[1]["system"]
        
        # Should be just the base system prompt
        assert system_content == generator.SYSTEM_PROMPT
        assert "Previous conversation" not in system_content