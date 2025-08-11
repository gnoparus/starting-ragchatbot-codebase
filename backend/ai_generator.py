import time
from typing import Any, Dict, List, Optional, Tuple

import anthropic
from logger import get_logger, log_execution_time

# Initialize logger for this module
log = get_logger("ai_generator")


class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""

    def __init__(self, api_key: str, model: str):
        log.info("Initializing AI Generator", model=model)
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        log.success("AI Generator initialized successfully")

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to comprehensive search and outline tools for course information.

Tool Usage Guidelines:
- **Sequential tool usage**: You can make up to 2 tool calls to gather comprehensive information
- **Round 1**: Use tools to gather initial information based on the user's query  
- **Round 2**: If needed, use tools again to gather additional information or compare sources
- **Content searches**: Use search tool for questions about specific course content or detailed educational materials
- **Course outlines**: Use outline tool for questions about course structure, syllabi, lesson lists, or course overviews
- **Complex queries**: Break down multi-part questions using multiple tool calls
- **Comparisons**: Use separate searches to compare information across courses or lessons
- Synthesize tool results into accurate, fact-based responses
- If tools yield no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without using tools
- **Course-specific content questions**: Use search tool first, then answer
- **Course outline/structure questions**: Use outline tool first, then answer
- **Multi-round reasoning**: Use tool results from previous rounds to inform subsequent searches
- **Final synthesis**: Combine all tool results into a comprehensive, accurate response
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, tool explanations, or question-type analysis
 - Do not mention "based on the search results" or "based on the outline"

Outline Query Handling:
When users ask about course structure, outline, syllabus, or lesson organization:
- Use the outline tool to get complete course information
- Present the course title, instructor, course link, and complete lesson list
- Format lesson information clearly with lesson numbers and titles
- Include lesson links when available

All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""

    def __init__(self, api_key: str, model: str):
        log.info("Initializing AI Generator", model=model)
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

        # Pre-build base API parameters
        self.base_params = {"model": self.model, "temperature": 0, "max_tokens": 800}
        log.success("AI Generator initialized successfully")

    @log_execution_time("generate_response")
    def generate_response(
        self,
        query: str,
        conversation_history: Optional[str] = None,
        tools: Optional[List] = None,
        tool_manager=None,
    ) -> str:
        """
        Generate AI response with optional tool usage and conversation context.
        Supports sequential tool calling for complex queries requiring multiple searches.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools

        Returns:
            Generated response as string
        """

        log.info(
            "Generating AI response",
            query_length=len(query),
            has_history=bool(conversation_history),
            tools_available=len(tools) if tools else 0,
            has_tool_manager=bool(tool_manager),
            query_preview=query[:100] + "..." if len(query) > 100 else query,
        )

        # Build system content efficiently
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        # Initialize conversation with user query
        initial_messages = [{"role": "user", "content": query}]

        # Use sequential tool calling if tools and tool manager are available
        if tools and tool_manager:
            log.debug(
                "Using sequential tool calling approach",
                max_rounds=2,
                tools_count=len(tools),
            )
            return self._execute_tool_rounds(
                messages=initial_messages,
                system_content=system_content,
                tools=tools,
                tool_manager=tool_manager,
                max_rounds=2,
            )
        else:
            log.debug("Using direct response approach", has_tools=bool(tools))
            # Direct response - include tools even if no tool_manager for compatibility
            api_params = {
                **self.base_params,
                "messages": initial_messages,
                "system": system_content,
            }

            # Add tools if available (for backwards compatibility)
            if tools:
                api_params["tools"] = tools
                api_params["tool_choice"] = {"type": "auto"}
                log.debug(f"Added {len(tools)} tools to API call")

            try:
                log.debug(
                    "Calling Anthropic API",
                    model=self.model,
                    temperature=self.base_params["temperature"],
                )
                response = self.client.messages.create(**api_params)

                log.success(
                    "Received API response",
                    stop_reason=response.stop_reason,
                    content_length=len(response.content) if response.content else 0,
                    content_types=(
                        [c.type for c in response.content] if response.content else []
                    ),
                    usage_input_tokens=(
                        response.usage.input_tokens if response.usage else None
                    ),
                    usage_output_tokens=(
                        response.usage.output_tokens if response.usage else None
                    ),
                )

                # Handle case where tools are used but no tool_manager provided
                if response.stop_reason == "tool_use" and not tool_manager:
                    log.warning("Tool use requested but no tool manager provided")
                    return "Error: Tool use requested but no tool manager provided"

                # Check if content is available
                if not response.content or len(response.content) == 0:
                    log.error("No content in API response")
                    return "Error: Empty response from AI service"

                return response.content[0].text

            except Exception as e:
                log.error(
                    "Error in direct API call",
                    error=str(e),
                    error_type=type(e).__name__,
                    query_preview=query[:100] + "..." if len(query) > 100 else query,
                )
                raise

    def _handle_tool_execution(
        self, initial_response, base_params: Dict[str, Any], tool_manager
    ):
        """
        Handle execution of tool calls and get follow-up response.

        Args:
            initial_response: The response containing tool use requests
            base_params: Base API parameters
            tool_manager: Manager to execute tools

        Returns:
            Final response text after tool execution
        """
        # Start with existing messages
        messages = base_params["messages"].copy()

        # Add AI's tool use response
        messages.append({"role": "assistant", "content": initial_response.content})

        # Execute all tool calls and collect results
        tool_results = []
        for content_block in initial_response.content:
            if content_block.type == "tool_use":
                tool_result = tool_manager.execute_tool(
                    content_block.name, **content_block.input
                )

                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": content_block.id,
                        "content": tool_result,
                    }
                )

        # Add tool results as single message
        if tool_results:
            messages.append({"role": "user", "content": tool_results})

        # Prepare final API call without tools
        final_params = {
            **self.base_params,
            "messages": messages,
            "system": base_params["system"],
        }

        # Get final response
        final_response = self.client.messages.create(**final_params)

        # Check if content is available
        if not final_response.content or len(final_response.content) == 0:
            return "Error: Empty response from AI service"

        return final_response.content[0].text

    def _execute_tool_rounds(
        self,
        messages: List,
        system_content: str,
        tools: List,
        tool_manager,
        max_rounds: int = 2,
    ) -> str:
        """
        Execute up to max_rounds of tool calling with Claude.

        Args:
            messages: Initial conversation messages
            system_content: System prompt content
            tools: Available tools for Claude
            tool_manager: Manager to execute tools
            max_rounds: Maximum number of tool calling rounds

        Returns:
            Final response after all rounds completed
        """
        current_messages = messages.copy()

        for round_num in range(1, max_rounds + 1):
            should_continue, current_messages = self._execute_single_round(
                messages=current_messages,
                system_content=system_content,
                tools=tools,
                tool_manager=tool_manager,
                round_num=round_num,
            )

            if not should_continue:
                break

        # Generate final response without tools
        return self._generate_final_response(current_messages, system_content)

    def _execute_single_round(
        self,
        messages: List,
        system_content: str,
        tools: List,
        tool_manager,
        round_num: int,
    ) -> Tuple[bool, List]:
        """
        Execute a single round of tool calling.

        Args:
            messages: Current conversation messages
            system_content: System prompt content
            tools: Available tools for Claude
            tool_manager: Manager to execute tools
            round_num: Current round number

        Returns:
            Tuple of (should_continue, updated_messages)
        """
        try:
            # Prepare API call with tools
            api_params = {
                **self.base_params,
                "messages": messages,
                "system": system_content,
                "tools": tools,
                "tool_choice": {"type": "auto"},
            }

            # Get response from Claude
            response = self.client.messages.create(**api_params)

            # Check if Claude used tools
            if response.stop_reason != "tool_use":
                # No tool use - we can stop here and use this as final response
                return False, messages

            # Add Claude's response to conversation
            messages.append({"role": "assistant", "content": response.content})

            # Execute all tool calls and collect results
            tool_results = []
            for content_block in response.content:
                if content_block.type == "tool_use":
                    try:
                        tool_result = tool_manager.execute_tool(
                            content_block.name, **content_block.input
                        )
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": content_block.id,
                                "content": tool_result,
                            }
                        )
                    except Exception as e:
                        # Handle tool execution errors gracefully
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": content_block.id,
                                "content": f"Tool execution error: {str(e)}",
                            }
                        )
                        # Stop on tool errors
                        return False, messages

            # Add tool results to conversation
            if tool_results:
                messages.append({"role": "user", "content": tool_results})

            # Continue to next round
            return True, messages

        except Exception as e:
            # Handle API errors - stop execution
            log.error(
                f"Anthropic API error in round {round_num}",
                round_num=round_num,
                error=str(e),
                error_type=type(e).__name__,
                query_preview=query[:100] + "..." if len(query) > 100 else query,
            )
            return False, messages

    def _generate_final_response(self, messages: List, system_content: str) -> str:
        """
        Generate final response without tools after all rounds completed.

        Args:
            messages: Full conversation history including tool results
            system_content: System prompt content

        Returns:
            Final synthesized response
        """
        try:
            # Final API call without tools for synthesis
            final_params = {
                **self.base_params,
                "messages": messages,
                "system": system_content,
            }

            log.debug(
                "Making final API call",
                messages_count=len(messages),
                system_length=len(system_content),
                params_keys=list(final_params.keys()),
            )

            final_response = self.client.messages.create(**final_params)

            log.debug(
                "Final API response received",
                stop_reason=final_response.stop_reason,
                content_length=(
                    len(final_response.content) if final_response.content else 0
                ),
                content_types=(
                    [c.type for c in final_response.content]
                    if final_response.content
                    else []
                ),
                usage_input=(
                    final_response.usage.input_tokens if final_response.usage else None
                ),
                usage_output=(
                    final_response.usage.output_tokens if final_response.usage else None
                ),
            )

            # Check if content is available
            if not final_response.content or len(final_response.content) == 0:
                log.error("No content in final API response")
                return "Error: Empty response from AI service"

            return final_response.content[0].text

        except Exception as e:
            log.error(
                "Exception in final response generation",
                error=str(e),
                error_type=type(e).__name__,
            )
            return f"Error generating final response: {str(e)}"
