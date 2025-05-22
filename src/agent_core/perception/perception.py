"""
Perception module for the TAAT AI Agent.

This module handles input processing and environment sensing for the agent,
including parsing user input and monitoring external data sources.
"""

from typing import Any, Dict, Optional


class PerceptionModule:
    """
    Perception module for the TAAT AI Agent.
    
    Handles processing of inputs from various sources, including user messages,
    social media data, and market information.
    """
    
    def __init__(self):
        """Initialize the perception module."""
        self.input_processors = {}
        self._register_default_processors()
    
    def _register_default_processors(self) -> None:
        """Register the default input processors."""
        # Register the text input processor by default
        self.register_processor("text", self._process_text_input)
    
    def register_processor(self, input_type: str, processor_func) -> None:
        """
        Register a new input processor.
        
        Args:
            input_type: Type of input this processor handles
            processor_func: Function that processes this input type
        """
        self.input_processors[input_type] = processor_func
    
    async def process_input(self, input_data: Any, input_type: str = "text") -> Dict[str, Any]:
        """
        Process input data using the appropriate processor.
        
        Args:
            input_data: The input data to process
            input_type: The type of input (default: "text")
            
        Returns:
            Processed input data
            
        Raises:
            ValueError: If no processor is registered for the input type
        """
        if input_type not in self.input_processors:
            raise ValueError(f"No processor registered for input type: {input_type}")
        
        processor = self.input_processors[input_type]
        return await processor(input_data)
    
    async def _process_text_input(self, text: str) -> Dict[str, Any]:
        """
        Process text input.
        
        Args:
            text: The text input to process
            
        Returns:
            Processed input data
        """
        # For now, just return the text with some basic metadata
        return {
            "type": "text",
            "content": text,
            "metadata": {
                "length": len(text),
                "word_count": len(text.split())
            }
        }
