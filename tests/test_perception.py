"""
Tests for the perception module.

This module contains tests for the PerceptionModule class.
"""

import pytest
import asyncio

from src.agent_core.perception.perception import PerceptionModule


@pytest.mark.asyncio
async def test_perception_module_initialization():
    """Test that PerceptionModule initializes correctly."""
    perception = PerceptionModule()
    assert "text" in perception.input_processors
    assert callable(perception.input_processors["text"])


@pytest.mark.asyncio
async def test_process_text_input():
    """Test processing of text input."""
    perception = PerceptionModule()
    result = await perception.process_input("Hello, world!", "text")
    
    assert result["type"] == "text"
    assert result["content"] == "Hello, world!"
    assert "metadata" in result
    assert result["metadata"]["length"] == 13
    assert result["metadata"]["word_count"] == 2


@pytest.mark.asyncio
async def test_register_custom_processor():
    """Test registering and using a custom processor."""
    perception = PerceptionModule()
    
    # Define a custom processor
    async def process_json(data):
        return {
            "type": "json",
            "content": data,
            "metadata": {
                "keys": list(data.keys()) if isinstance(data, dict) else []
            }
        }
    
    # Register the processor
    perception.register_processor("json", process_json)
    
    # Test the processor
    test_data = {"key": "value"}
    result = await perception.process_input(test_data, "json")
    
    assert result["type"] == "json"
    assert result["content"] == test_data
    assert result["metadata"]["keys"] == ["key"]


@pytest.mark.asyncio
async def test_invalid_input_type():
    """Test handling of invalid input type."""
    perception = PerceptionModule()
    
    with pytest.raises(ValueError):
        await perception.process_input("test", "invalid_type")
