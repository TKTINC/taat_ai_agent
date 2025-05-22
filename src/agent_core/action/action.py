"""
Action module for the TAAT AI Agent.

This module handles execution of decisions made by the cognition module,
including communication with users and external systems.
"""

from typing import Any, Dict, Optional, Callable, List


class ToolRegistry:
    """
    Registry for tools that the agent can use to take actions.
    """
    
    def __init__(self):
        """Initialize the tool registry."""
        self.tools: Dict[str, Dict[str, Any]] = {}
    
    def register_tool(self, name: str, func: Callable, description: str) -> None:
        """
        Register a new tool.
        
        Args:
            name: Name of the tool
            func: Function that implements the tool
            description: Description of what the tool does
        """
        self.tools[name] = {
            "func": func,
            "description": description
        }
    
    def get_tool(self, name: str) -> Optional[Callable]:
        """
        Get a tool by name.
        
        Args:
            name: Name of the tool
            
        Returns:
            The tool function or None if not found
        """
        tool = self.tools.get(name)
        return tool["func"] if tool else None
    
    def get_tool_descriptions(self) -> List[Dict[str, str]]:
        """
        Get descriptions of all registered tools.
        
        Returns:
            List of tool descriptions
        """
        return [
            {"name": name, "description": details["description"]}
            for name, details in self.tools.items()
        ]


class ActionModule:
    """
    Action module for the TAAT AI Agent.
    
    Handles execution of decisions made by the cognition module,
    including communication with users and external systems.
    """
    
    def __init__(self):
        """Initialize the action module."""
        self.tool_registry = ToolRegistry()
        self._register_default_tools()
    
    def _register_default_tools(self) -> None:
        """Register the default tools."""
        self.tool_registry.register_tool(
            "send_message",
            self._send_message,
            "Send a message to the user"
        )
    
    async def _send_message(self, message: str) -> Dict[str, Any]:
        """
        Send a message to the user.
        
        Args:
            message: The message to send
            
        Returns:
            Result of the action
        """
        # In a real implementation, this would send the message through some interface
        print(f"AGENT: {message}")
        return {
            "status": "success",
            "message": message
        }
    
    async def execute(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute an action based on the cognition module's response.
        
        Args:
            response: The response from the cognition module
            
        Returns:
            Result of the action
        """
        # For now, just handle text responses by sending them as messages
        if response.get("type") == "text":
            return await self._send_message(response["content"])
        
        # In the future, this will handle tool calls and other action types
        return {
            "status": "error",
            "message": f"Unsupported response type: {response.get('type')}"
        }
