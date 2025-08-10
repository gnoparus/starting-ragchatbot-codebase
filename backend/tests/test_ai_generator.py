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
        """Test response generation with tool use (updated for sequential behavior)"""
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        
        # Mock early termination after first tool use
        mock_non_tool_response = Mock()
        mock_non_tool_response.stop_reason = "end_turn"
        mock_non_tool_response.content = [Mock(text="No more tools needed")]
        
        # Sequential tool calling: round 1 tool use, round 2 no tools, final synthesis
        mock_client.messages.create.side_effect = [mock_tool_use_response, mock_non_tool_response, mock_final_response]
        
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
        
        # Check that three API calls were made (sequential tool calling)
        assert mock_client.messages.create.call_count == 3

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

    # Sequential Tool Calling Tests
    
    @patch('ai_generator.anthropic.Anthropic')
    def test_sequential_tool_calling_two_rounds_complete(self, mock_anthropic_class, sample_tool_definitions):
        """Test complete 2-round sequential tool calling"""
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        
        # Round 1: Tool use response
        round1_response = Mock()
        round1_response.stop_reason = "tool_use"
        round1_tool_block = Mock()
        round1_tool_block.type = "tool_use"
        round1_tool_block.name = "search_course_content"
        round1_tool_block.id = "tool_round1"
        round1_tool_block.input = {"query": "round 1 query"}
        round1_response.content = [round1_tool_block]
        
        # Round 2: Another tool use response
        round2_response = Mock()
        round2_response.stop_reason = "tool_use"
        round2_tool_block = Mock()
        round2_tool_block.type = "tool_use"
        round2_tool_block.name = "search_course_content"
        round2_tool_block.id = "tool_round2"
        round2_tool_block.input = {"query": "round 2 query"}
        round2_response.content = [round2_tool_block]
        
        # Final response (no tools)
        final_response = Mock()
        final_response.content = [Mock(text="Final answer after 2 rounds")]
        
        # Mock 3 API calls: round1 -> round2 -> final
        mock_client.messages.create.side_effect = [round1_response, round2_response, final_response]
        
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = [
            "Round 1 tool result", 
            "Round 2 tool result"
        ]
        
        generator = AIGenerator("test-key", "test-model")
        
        result = generator.generate_response(
            "Compare topics across courses",
            tools=sample_tool_definitions,
            tool_manager=mock_tool_manager
        )
        
        # Verify exactly 3 API calls were made
        assert mock_client.messages.create.call_count == 3
        
        # Verify 2 tool executions
        assert mock_tool_manager.execute_tool.call_count == 2
        mock_tool_manager.execute_tool.assert_any_call("search_course_content", query="round 1 query")
        mock_tool_manager.execute_tool.assert_any_call("search_course_content", query="round 2 query")
        
        # Verify final result
        assert result == "Final answer after 2 rounds"
        
        # Verify message history structure in final call
        final_call_args = mock_client.messages.create.call_args_list[2]
        final_messages = final_call_args[1]["messages"]
        
        # Should have: user_query + assistant_round1 + user_tool_results + assistant_round2 + user_tool_results
        assert len(final_messages) == 5
        assert final_messages[0]["role"] == "user"  # Original query
        assert final_messages[1]["role"] == "assistant"  # Round 1 response
        assert final_messages[2]["role"] == "user"  # Round 1 tool results
        assert final_messages[3]["role"] == "assistant"  # Round 2 response  
        assert final_messages[4]["role"] == "user"  # Round 2 tool results

    @patch('ai_generator.anthropic.Anthropic')
    def test_sequential_early_termination_on_non_tool_response(self, mock_anthropic_class, sample_tool_definitions):
        """Test termination when first round doesn't use tools"""
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        
        # First response has no tool use
        non_tool_response = Mock()
        non_tool_response.stop_reason = "end_turn"
        non_tool_response.content = [Mock(text="Direct answer without tools")]
        
        # Final response
        final_response = Mock()
        final_response.content = [Mock(text="Direct answer without tools")]
        
        mock_client.messages.create.side_effect = [non_tool_response, final_response]
        
        generator = AIGenerator("test-key", "test-model")
        result = generator.generate_response(
            "simple query",
            tools=sample_tool_definitions,
            tool_manager=Mock()
        )
        
        # Should make 2 API calls: initial round + final synthesis
        assert mock_client.messages.create.call_count == 2
        assert result == "Direct answer without tools"

    @patch('ai_generator.anthropic.Anthropic')
    def test_sequential_termination_on_tool_failure(self, mock_anthropic_class, sample_tool_definitions):
        """Test termination when tool execution fails"""
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        
        # Round 1: Tool use response
        tool_use_response = Mock()
        tool_use_response.stop_reason = "tool_use"
        tool_block = Mock()
        tool_block.type = "tool_use"
        tool_block.name = "search_course_content"
        tool_block.id = "tool_123"
        tool_block.input = {"query": "test query"}
        tool_use_response.content = [tool_block]
        
        # Final response after tool error
        final_response = Mock()
        final_response.content = [Mock(text="Response after tool error")]
        
        mock_client.messages.create.side_effect = [tool_use_response, final_response]
        
        # Tool manager raises exception
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = Exception("Tool execution failed")
        
        generator = AIGenerator("test-key", "test-model")
        
        result = generator.generate_response(
            "test query",
            tools=sample_tool_definitions,
            tool_manager=mock_tool_manager
        )
        
        # Should make 2 API calls: initial round with tool failure + final synthesis
        assert mock_client.messages.create.call_count == 2
        assert result == "Response after tool error"

    @patch('ai_generator.anthropic.Anthropic')
    def test_sequential_conversation_context_preserved(self, mock_anthropic_class, sample_tool_definitions):
        """Test that conversation context is preserved across rounds"""
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        
        # Single round with tool use
        tool_response = Mock()
        tool_response.stop_reason = "tool_use"
        tool_block = Mock()
        tool_block.type = "tool_use"
        tool_block.name = "search_course_content"
        tool_block.id = "tool_123"
        tool_block.input = {"query": "search query"}
        tool_response.content = [tool_block]
        
        # Final response
        final_response = Mock()
        final_response.content = [Mock(text="Answer with context")]
        
        mock_client.messages.create.side_effect = [tool_response, final_response]
        
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Tool result"
        
        generator = AIGenerator("test-key", "test-model")
        
        # Include conversation history
        history = "Previous: User asked about AI, I explained basics"
        result = generator.generate_response(
            "Follow up question",
            conversation_history=history,
            tools=sample_tool_definitions,
            tool_manager=mock_tool_manager
        )
        
        # Check that history is included in system prompt for all calls
        for call_args in mock_client.messages.create.call_args_list:
            system_content = call_args[1]["system"]
            assert "Previous conversation" in system_content
            assert history in system_content

    @patch('ai_generator.anthropic.Anthropic')
    def test_sequential_max_rounds_enforcement(self, mock_anthropic_class, sample_tool_definitions):
        """Test that system enforces maximum of 2 rounds"""
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        
        # Create tool use responses for more than 2 rounds
        def create_tool_response(round_num):
            response = Mock()
            response.stop_reason = "tool_use"
            tool_block = Mock()
            tool_block.type = "tool_use"
            tool_block.name = "search_course_content"
            tool_block.id = f"tool_round_{round_num}"
            tool_block.input = {"query": f"round {round_num} query"}
            response.content = [tool_block]
            return response
        
        # Final response
        final_response = Mock()
        final_response.content = [Mock(text="Final answer after max rounds")]
        
        # Mock responses: round1, round2, final (should not call round3)
        mock_client.messages.create.side_effect = [
            create_tool_response(1),
            create_tool_response(2), 
            final_response
        ]
        
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = [
            "Round 1 result",
            "Round 2 result"
        ]
        
        generator = AIGenerator("test-key", "test-model")
        
        result = generator.generate_response(
            "complex query needing multiple searches",
            tools=sample_tool_definitions,
            tool_manager=mock_tool_manager
        )
        
        # Should make exactly 3 API calls (2 rounds + final)
        assert mock_client.messages.create.call_count == 3
        
        # Should execute exactly 2 tools (max rounds)
        assert mock_tool_manager.execute_tool.call_count == 2
        
        assert result == "Final answer after max rounds"

    @patch('ai_generator.anthropic.Anthropic')
    def test_sequential_api_error_handling(self, mock_anthropic_class, sample_tool_definitions):
        """Test handling of API errors during sequential rounds"""
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        
        # First call succeeds, second call fails
        tool_response = Mock()
        tool_response.stop_reason = "tool_use"
        tool_block = Mock()
        tool_block.type = "tool_use"
        tool_block.name = "search_course_content"
        tool_block.id = "tool_123"
        tool_block.input = {"query": "test query"}
        tool_response.content = [tool_block]
        
        final_response = Mock()
        final_response.content = [Mock(text="Response after API error")]
        
        mock_client.messages.create.side_effect = [
            tool_response,
            Exception("API Error in round 2"),
            final_response
        ]
        
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Tool result"
        
        generator = AIGenerator("test-key", "test-model")
        
        result = generator.generate_response(
            "test query",
            tools=sample_tool_definitions,
            tool_manager=mock_tool_manager
        )
        
        # Should handle the API error gracefully
        assert result == "Response after API error"

    def test_system_prompt_sequential_updates(self):
        """Test that system prompt includes sequential tool calling guidance"""
        generator = AIGenerator("test-key", "test-model")
        
        # Check for sequential tool calling content
        assert "Sequential tool usage" in generator.SYSTEM_PROMPT
        assert "up to 2 tool calls" in generator.SYSTEM_PROMPT
        assert "Round 1" in generator.SYSTEM_PROMPT
        assert "Round 2" in generator.SYSTEM_PROMPT
        assert "Multi-round reasoning" in generator.SYSTEM_PROMPT
        
        # Should not have the old limitation
        assert "One tool call per query maximum" not in generator.SYSTEM_PROMPT