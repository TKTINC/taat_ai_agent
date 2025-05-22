"""
Memory module for the TAAT AI Agent.

This module handles working memory and context management for the agent,
including conversation history and state tracking.
"""

from datetime import datetime
from typing import Dict, List, Any, Optional


class WorkingMemory:
    """
    Working memory system for the TAAT AI Agent.
    
    Handles conversation history and state tracking to maintain context
    across interactions.
    """
    
    def __init__(self, max_history: int = 10):
        """
        Initialize the working memory.
        
        Args:
            max_history: Maximum number of conversation turns to keep in history
        """
        self.conversation_history: List[Dict[str, Any]] = []
        self.max_history = max_history
        self.state: Dict[str, Any] = {}
    
    def get_context(self) -> Dict[str, Any]:
        """
        Get the current context for decision-making.
        
        Returns:
            Dict containing conversation history and current state
        """
        return {
            "conversation": self.conversation_history,
            "state": self.state
        }
    
    def update(self, input_data: Any, response: Any, result: Any) -> None:
        """
        Update the memory with a new interaction.
        
        Args:
            input_data: The input received by the agent
            response: The agent's response
            result: The result of executing the response
        """
        self.conversation_history.append({
            "input": input_data,
            "response": response,
            "result": result,
            "timestamp": datetime.now().isoformat()
        })
        
        # Trim history if needed
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]
    
    def set_state(self, key: str, value: Any) -> None:
        """
        Set a value in the agent's state.
        
        Args:
            key: State key
            value: State value
        """
        self.state[key] = value
    
    def get_state(self, key: str, default: Optional[Any] = None) -> Any:
        """
        Get a value from the agent's state.
        
        Args:
            key: State key
            default: Default value if key doesn't exist
            
        Returns:
            The state value or default
        """
        return self.state.get(key, default)
    
    def clear_state(self) -> None:
        """Clear the entire state."""
        self.state = {}
    
    def clear_history(self) -> None:
        """Clear the conversation history."""
        self.conversation_history = []
    
    def reset(self) -> None:
        """Reset both state and history."""
        self.clear_state()
        self.clear_history()
