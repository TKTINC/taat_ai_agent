"""
Tests for the cognition module.

This module contains tests for the CognitionModule class.
"""

import pytest
import asyncio
from unittest.mock import patch, MagicMock

from src.agent_core.cognition.cognition import CognitionModule


@pytest.fixture
def mock_anthropic_client():
    """Fixture for mocking the Anthropic client."""
    with patch('anthropic.Anthropic') as mock_client:
        # Create a mock response
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Mock response")]
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 20
        
        # Set up the mock client to return the mock response
        mock_client.return_value.messages.create = MagicMock(
            return_value=mock_response
        )
        
        yield mock_client


@pytest.mark.asyncio
async def test_cognition_module_initialization(mock_llm_settings):
    """Test that CognitionModule initializes correctly."""
    with patch('anthropic.Anthropic'):
        cognition = CognitionModule(mock_llm_settings)
        assert cognition.llm_settings == mock_llm_settings
        assert cognition.system_prompt is not None


@pytest.mark.asyncio
async def test_set_system_prompt(mock_llm_settings):
    """Test setting a custom system prompt."""
    with patch('anthropic.Anthropic'):
        cognition = CognitionModule(mock_llm_settings)
        custom_prompt = "Custom system prompt"
        cognition.set_system_prompt(custom_prompt)
        assert cognition.system_prompt == custom_prompt


@pytest.mark.asyncio
async def test_format_messages(mock_llm_settings):
    """Test formatting of messages for the LLM."""
    with patch('anthropic.Anthropic'):
        cognition = CognitionModule(mock_llm_settings)
        
        # Create test data
        context = {
            "conversation": [
                {
                    "input": {"content": "User message 1"},
                    "response": {"content": "Assistant response 1"}
                }
            ]
        }
        input_data = {"content": "User message 2"}
        
        # Format messages
        messages = cognition._format_messages(context, input_data)
        
        # Verify the result
        assert len(messages) == 3
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "User message 1"
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "Assistant response 1"
        assert messages[2]["role"] == "user"
        assert messages[2]["content"] == "User message 2"


@pytest.mark.asyncio
async def test_process(mock_llm_settings, mock_anthropic_client):
    """Test processing input and generating a response."""
    cognition = CognitionModule(mock_llm_settings)
    
    # Create test data
    context = {"conversation": []}
    input_data = {"content": "Test input"}
    
    # Process the input
    response = await cognition.process(input_data, context)
    
    # Verify the response
    assert response["type"] == "text"
    assert response["content"] == "Mock response"
    assert "metadata" in response
    assert response["metadata"]["model"] == mock_llm_settings.model
    assert response["metadata"]["usage"]["input_tokens"] == 10
    assert response["metadata"]["usage"]["output_tokens"] == 20
    
    # Verify the client was called correctly
    mock_anthropic_client.return_value.messages.create.assert_called_once()
    call_args = mock_anthropic_client.return_value.messages.create.call_args[1]
    assert call_args["model"] == mock_llm_settings.model
    assert call_args["system"] == cognition.system_prompt
    assert isinstance(call_args["messages"], list)
