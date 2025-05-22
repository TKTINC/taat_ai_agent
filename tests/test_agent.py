"""
Tests for the main agent class.

This module contains tests for the TaatAgent class.
"""

import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock

from src.agent_core.agent import TaatAgent
from src.agent_core.config import AgentConfig


@pytest.fixture
def mock_components():
    """Fixture for mocking all agent components."""
    with patch('src.agent_core.memory.memory.WorkingMemory') as mock_memory, \
         patch('src.agent_core.perception.perception.PerceptionModule') as mock_perception, \
         patch('src.agent_core.cognition.cognition.CognitionModule') as mock_cognition, \
         patch('src.agent_core.action.action.ActionModule') as mock_action:
        
        # Set up mock methods
        mock_memory.return_value.get_context.return_value = {"mock": "context"}
        mock_memory.return_value.update = MagicMock()
        
        mock_perception.return_value.process_input = AsyncMock(
            return_value={"type": "text", "content": "Processed input"}
        )
        
        mock_cognition.return_value.process = AsyncMock(
            return_value={"type": "text", "content": "Cognition response"}
        )
        
        mock_action.return_value.execute = AsyncMock(
            return_value={"status": "success", "message": "Action result"}
        )
        
        yield {
            "memory": mock_memory,
            "perception": mock_perception,
            "cognition": mock_cognition,
            "action": mock_action
        }


@pytest.mark.asyncio
async def test_agent_initialization(mock_agent_config, mock_components):
    """Test that TaatAgent initializes correctly."""
    with patch('src.agent_core.agent.load_config', return_value=mock_agent_config):
        agent = TaatAgent()
        
        # Verify components were initialized
        mock_components["memory"].assert_called_once()
        mock_components["perception"].assert_called_once()
        mock_components["cognition"].assert_called_once_with(mock_agent_config.llm_settings)
        mock_components["action"].assert_called_once()
        
        # Verify initial state
        assert agent.running == False


@pytest.mark.asyncio
async def test_process_input(mock_agent_config, mock_components):
    """Test the process_input method."""
    with patch('src.agent_core.agent.load_config', return_value=mock_agent_config):
        agent = TaatAgent()
        
        # Process an input
        result = await agent.process_input("Test input")
        
        # Verify each component was called correctly
        mock_components["perception"].return_value.process_input.assert_called_once_with(
            "Test input", "text"
        )
        
        mock_components["memory"].return_value.get_context.assert_called_once()
        
        mock_components["cognition"].return_value.process.assert_called_once_with(
            {"type": "text", "content": "Processed input"},
            {"mock": "context"}
        )
        
        mock_components["action"].return_value.execute.assert_called_once_with(
            {"type": "text", "content": "Cognition response"}
        )
        
        # Verify memory was updated
        mock_components["memory"].return_value.update.assert_called_once_with(
            {"type": "text", "content": "Processed input"},
            {"type": "text", "content": "Cognition response"},
            {"status": "success", "message": "Action result"}
        )
        
        # Verify the result
        assert result == {"status": "success", "message": "Action result"}


@pytest.mark.asyncio
async def test_run_loop(mock_agent_config, mock_components):
    """Test the run_loop method."""
    with patch('src.agent_core.agent.load_config', return_value=mock_agent_config), \
         patch('builtins.input', side_effect=["Test input", "exit"]), \
         patch('builtins.print'):
        
        agent = TaatAgent()
        
        # Run the loop
        await agent.run_loop()
        
        # Verify process_input was called
        mock_components["perception"].return_value.process_input.assert_called_once()
        
        # Verify the agent stopped
        assert agent.running == False


def test_stop(mock_agent_config, mock_components):
    """Test the stop method."""
    with patch('src.agent_core.agent.load_config', return_value=mock_agent_config):
        agent = TaatAgent()
        agent.running = True
        
        # Stop the agent
        agent.stop()
        
        # Verify the agent stopped
        assert agent.running == False
