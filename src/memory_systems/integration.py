"""
Integration with the core agent architecture.

This module provides integration between the core agent architecture and
the advanced memory systems.
"""

import asyncio
from typing import Dict, List, Any, Optional

from src.agent_core.agent import TaatAgent
from src.memory_systems.config import load_enhanced_config
from src.memory_systems.manager import MemoryManager


class EnhancedTaatAgent(TaatAgent):
    """
    Enhanced TAAT Agent with advanced memory systems.
    
    Extends the core TaatAgent with episodic, semantic, and procedural memory.
    """
    
    def __init__(self, config=None):
        """
        Initialize the enhanced TAAT Agent.
        
        Args:
            config: Agent configuration (loads from environment if None)
        """
        # Load enhanced configuration
        self.enhanced_config = config or load_enhanced_config()
        
        # Initialize base agent with enhanced config
        super().__init__(self.enhanced_config)
        
        # Replace basic working memory with memory manager
        self.memory_manager = MemoryManager(self.enhanced_config.memory)
        
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
        
        # 2. Get enhanced context from memory systems
        context = await self.memory_manager.get_context(processed_input)
        
        # 3. Cognition: Generate response
        response = await self.cognition.process(processed_input, context)
        
        # 4. Action: Execute response
        result = await self.action.execute(response)
        
        # 5. Memory: Update all memory systems
        await self.memory_manager.update_memories(processed_input, response, result)
        
        return result
