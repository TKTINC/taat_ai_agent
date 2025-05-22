"""
Main agent class for the TAAT AI Agent.

This module contains the main TaatAgent class that integrates all components
and implements the perception-cognition-action loop.
"""

import asyncio
from typing import Any, Dict, Optional

from .config import AgentConfig, load_config
from .memory.memory import WorkingMemory
from .perception.perception import PerceptionModule
from .cognition.cognition import CognitionModule
from .action.action import ActionModule, ToolRegistry


class TaatAgent:
    """
    Main agent class for the TAAT AI Agent.
    
    Integrates perception, cognition, action, and memory components
    and implements the perception-cognition-action loop.
    """
    
    def __init__(self, config: Optional[AgentConfig] = None):
        """
        Initialize the TAAT Agent.
        
        Args:
            config: Agent configuration (loads from environment if None)
        """
        self.config = config or load_config()
        self.memory = WorkingMemory(max_history=self.config.max_history)
        self.perception = PerceptionModule()
        self.cognition = CognitionModule(self.config.llm_settings)
        self.action = ActionModule()
        self.running = False
    
    async def process_input(self, input_data: Any, input_type: str = "text") -> Dict[str, Any]:
        """
        Process a single input through the perception-cognition-action loop.
        
        Args:
            input_data: The input data to process
            input_type: The type of input
            
        Returns:
            Result of the action
        """
        # 1. Perception: Process input
        processed_input = await self.perception.process_input(input_data, input_type)
        
        # 2. Cognition: Generate response
        context = self.memory.get_context()
        response = await self.cognition.process(processed_input, context)
        
        # 3. Action: Execute response
        result = await self.action.execute(response)
        
        # 4. Memory: Update with this interaction
        self.memory.update(processed_input, response, result)
        
        return result
    
    async def run_loop(self):
        """
        Run the agent's main loop, processing inputs continuously.
        
        This is a simple implementation that reads from stdin.
        In a real application, this would be replaced with a proper interface.
        """
        self.running = True
        print("TAAT Agent is running. Type 'exit' to quit.")
        
        while self.running:
            try:
                # Get input from user
                user_input = input("USER: ")
                
                if user_input.lower() == "exit":
                    self.running = False
                    print("TAAT Agent shutting down.")
                    break
                
                # Process the input
                await self.process_input(user_input)
                
            except KeyboardInterrupt:
                self.running = False
                print("\nTAAT Agent shutting down.")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    def stop(self):
        """Stop the agent's main loop."""
        self.running = False
