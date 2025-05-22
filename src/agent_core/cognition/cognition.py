"""
Cognition module for the TAAT AI Agent.

This module handles decision-making using the Claude LLM, including
processing inputs, generating responses, and making trade decisions.
"""

import anthropic
from typing import Any, Dict, List, Optional


class CognitionModule:
    """
    Cognition module for the TAAT AI Agent.
    
    Handles decision-making using the Claude LLM, including processing inputs,
    generating responses, and making trade decisions.
    """
    
    def __init__(self, llm_settings):
        """
        Initialize the cognition module.
        
        Args:
            llm_settings: Configuration for the LLM
        """
        self.llm_settings = llm_settings
        self.client = anthropic.Anthropic(api_key=llm_settings.api_key)
        self.system_prompt = self._get_default_system_prompt()
    
    def _get_default_system_prompt(self) -> str:
        """
        Get the default system prompt for the agent.
        
        Returns:
            Default system prompt
        """
        return """
        You are TAAT (Twitter Trade Announcer Tool), an AI Agent designed to monitor trader accounts on X (Twitter),
        identify trade signals from natural language posts, and assist with trade execution.
        
        Your primary goals are:
        1. Accurately identify trade signals from trader posts
        2. Extract key parameters (symbol, action, price points, etc.)
        3. Evaluate signals against user preferences
        4. Provide clear explanations for your decisions
        5. Learn from outcomes to improve over time
        
        You should be:
        - Precise in your analysis of trade signals
        - Transparent about your confidence levels
        - Cautious with ambiguous signals
        - Responsive to user feedback
        - Clear and concise in your communications
        
        You have access to various tools that will be provided through function calling.
        Always use the appropriate tool for each task.
        """
    
    def set_system_prompt(self, prompt: str) -> None:
        """
        Set a custom system prompt.
        
        Args:
            prompt: The new system prompt
        """
        self.system_prompt = prompt
    
    def _format_messages(self, context: Dict[str, Any], input_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Format the conversation history and current input into messages for the LLM.
        
        Args:
            context: The current context from working memory
            input_data: The current input data
            
        Returns:
            Formatted messages for the LLM
        """
        messages = []
        
        # Add conversation history
        for entry in context.get("conversation", []):
            if "input" in entry and isinstance(entry["input"], dict) and "content" in entry["input"]:
                messages.append({"role": "user", "content": entry["input"]["content"]})
            if "response" in entry and isinstance(entry["response"], dict) and "content" in entry["response"]:
                messages.append({"role": "assistant", "content": entry["response"]["content"]})
        
        # Add current input
        if "content" in input_data:
            messages.append({"role": "user", "content": input_data["content"]})
        
        return messages
    
    async def process(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input and generate a response using the LLM.
        
        Args:
            input_data: The input data to process
            context: The current context from working memory
            
        Returns:
            The LLM's response
        """
        messages = self._format_messages(context, input_data)
        
        response = await self.client.messages.create(
            model=self.llm_settings.model,
            system=self.system_prompt,
            messages=messages,
            max_tokens=self.llm_settings.max_tokens,
            temperature=self.llm_settings.temperature
        )
        
        return {
            "type": "text",
            "content": response.content[0].text,
            "metadata": {
                "model": self.llm_settings.model,
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens
                }
            }
        }
