"""
Tests for the memory module.

This module contains tests for the WorkingMemory class.
"""

import pytest
from datetime import datetime

from src.agent_core.memory.memory import WorkingMemory


def test_working_memory_initialization():
    """Test that WorkingMemory initializes correctly."""
    memory = WorkingMemory(max_history=5)
    assert memory.max_history == 5
    assert memory.conversation_history == []
    assert memory.state == {}


def test_get_context():
    """Test that get_context returns the correct structure."""
    memory = WorkingMemory()
    context = memory.get_context()
    assert "conversation" in context
    assert "state" in context
    assert context["conversation"] == []
    assert context["state"] == {}


def test_update_and_trim():
    """Test that update adds entries and trims history when needed."""
    memory = WorkingMemory(max_history=2)
    
    # Add first entry
    memory.update("input1", "response1", "result1")
    assert len(memory.conversation_history) == 1
    assert memory.conversation_history[0]["input"] == "input1"
    
    # Add second entry
    memory.update("input2", "response2", "result2")
    assert len(memory.conversation_history) == 2
    assert memory.conversation_history[1]["input"] == "input2"
    
    # Add third entry, should trim the first
    memory.update("input3", "response3", "result3")
    assert len(memory.conversation_history) == 2
    assert memory.conversation_history[0]["input"] == "input2"
    assert memory.conversation_history[1]["input"] == "input3"


def test_state_management():
    """Test state management functions."""
    memory = WorkingMemory()
    
    # Set and get state
    memory.set_state("key1", "value1")
    assert memory.get_state("key1") == "value1"
    
    # Get with default
    assert memory.get_state("nonexistent", "default") == "default"
    
    # Clear state
    memory.clear_state()
    assert memory.state == {}


def test_reset():
    """Test reset functionality."""
    memory = WorkingMemory()
    
    # Add some data
    memory.update("input", "response", "result")
    memory.set_state("key", "value")
    
    # Reset
    memory.reset()
    
    # Verify everything is cleared
    assert memory.conversation_history == []
    assert memory.state == {}
