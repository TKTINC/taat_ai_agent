"""
Configuration management for the TAAT AI Agent.

This module handles loading and validating configuration from environment variables
and configuration files.
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class LLMSettings:
    """Settings for the LLM integration."""
    api_key: str
    model: str = "claude-3-sonnet-20240229"
    max_tokens: int = 4096
    temperature: float = 0.7


@dataclass
class AgentConfig:
    """Main configuration for the TAAT AI Agent."""
    llm_settings: LLMSettings
    debug_mode: bool = False
    log_level: str = "INFO"
    max_history: int = 10


def load_config() -> AgentConfig:
    """
    Load configuration from environment variables.
    
    Returns:
        AgentConfig: The loaded configuration
        
    Raises:
        ValueError: If required environment variables are missing
    """
    # Check for required environment variables
    anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not anthropic_api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable is required")
    
    # Load LLM settings
    llm_settings = LLMSettings(
        api_key=anthropic_api_key,
        model=os.environ.get("LLM_MODEL", "claude-3-sonnet-20240229"),
        max_tokens=int(os.environ.get("LLM_MAX_TOKENS", "4096")),
        temperature=float(os.environ.get("LLM_TEMPERATURE", "0.7")),
    )
    
    # Load agent config
    debug_mode = os.environ.get("DEBUG_MODE", "False").lower() == "true"
    log_level = os.environ.get("LOG_LEVEL", "INFO")
    max_history = int(os.environ.get("MAX_HISTORY", "10"))
    
    return AgentConfig(
        llm_settings=llm_settings,
        debug_mode=debug_mode,
        log_level=log_level,
        max_history=max_history,
    )
