"""
Tests for the action module.

This module contains tests for the ActionModule and ToolRegistry classes.
"""

import pytest
import asyncio
from unittest.mock import patch, MagicMock

from src.agent_core.action.action import ActionModule, ToolRegistry


def test_tool_registry_initialization():
    """Test that ToolRegistry initializes correctly."""
    registry = ToolRegistry()
    assert registry.tools == {}


def test_register_and_get_tool():
    """Test registering and retrieving tools."""
    registry = ToolRegistry()
    
    # Define a test tool
    def test_tool(arg):
        return f"Test: {arg}"
    
    # Register the tool
    registry.register_tool("test", test_tool, "Test tool description")
    
    # Verify the tool was registered
    assert "test" in registry.tools
    assert registry.tools["test"]["func"] == test_tool
    assert registry.tools["test"]["description"] == "Test tool description"
    
    # Retrieve the tool
    tool_func = registry.get_tool("test")
    assert tool_func == test_tool
    assert tool_func("hello") == "Test: hello"
    
    # Try to get a non-existent tool
    assert registry.get_tool("nonexistent") is None


def test_get_tool_descriptions():
    """Test getting tool descriptions."""
    registry = ToolRegistry()
    
    # Register some tools
    registry.register_tool("tool1", lambda x: x, "Tool 1 description")
    registry.register_tool("tool2", lambda x: x, "Tool 2 description")
    
    # Get descriptions
    descriptions = registry.get_tool_descriptions()
    
    # Verify the descriptions
    assert len(descriptions) == 2
    assert {"name": "tool1", "description": "Tool 1 description"} in descriptions
    assert {"name": "tool2", "description": "Tool 2 description"} in descriptions


@pytest.mark.asyncio
async def test_action_module_initialization():
    """Test that ActionModule initializes correctly."""
    with patch('builtins.print'):
        action = ActionModule()
        assert isinstance(action.tool_registry, ToolRegistry)
        assert "send_message" in action.tool_registry.tools


@pytest.mark.asyncio
async def test_send_message():
    """Test the send_message tool."""
    with patch('builtins.print') as mock_print:
        action = ActionModule()
        result = await action._send_message("Test message")
        
        # Verify the message was printed
        mock_print.assert_called_once_with("AGENT: Test message")
        
        # Verify the result
        assert result["status"] == "success"
        assert result["message"] == "Test message"


@pytest.mark.asyncio
async def test_execute_text_response():
    """Test executing a text response."""
    with patch.object(ActionModule, '_send_message') as mock_send:
        mock_send.return_value = {"status": "success", "message": "Test"}
        
        action = ActionModule()
        response = {"type": "text", "content": "Test message"}
        result = await action.execute(response)
        
        # Verify send_message was called
        mock_send.assert_called_once_with("Test message")
        
        # Verify the result
        assert result["status"] == "success"


@pytest.mark.asyncio
async def test_execute_unsupported_response():
    """Test executing an unsupported response type."""
    action = ActionModule()
    response = {"type": "unsupported", "content": "Test"}
    result = await action.execute(response)
    
    # Verify the error result
    assert result["status"] == "error"
    assert "Unsupported response type" in result["message"]
