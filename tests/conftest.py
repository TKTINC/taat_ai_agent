"""
Test configuration for the TAAT AI Agent.

This module contains pytest configuration and fixtures.
"""

import os
import pytest
from dotenv import load_dotenv

from src.agent_core.config import AgentConfig, LLMSettings


@pytest.fixture
def mock_llm_settings():
    """Fixture for mock LLM settings."""
    return LLMSettings(
        api_key="mock_api_key",
        model="mock-model",
        max_tokens=100,
        temperature=0.5
    )


@pytest.fixture
def mock_agent_config(mock_llm_settings):
    """Fixture for mock agent configuration."""
    return AgentConfig(
        llm_settings=mock_llm_settings,
        debug_mode=True,
        log_level="DEBUG",
        max_history=5
    )
