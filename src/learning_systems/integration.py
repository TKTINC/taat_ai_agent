"""
Integration with the enhanced agent architecture.

This module provides integration between the enhanced agent architecture with memory systems
and the learning systems.
"""

import asyncio
from typing import Dict, List, Any, Optional

from src.memory_systems.integration import EnhancedTaatAgent
from src.learning_systems.config import load_learning_config
from src.learning_systems.manager import LearningManager


class LearningTaatAgent(EnhancedTaatAgent):
    """
    Learning TAAT Agent with advanced memory and learning systems.
    
    Extends the enhanced TaatAgent with reinforcement learning, feedback processing,
    performance tracking, and pattern recognition.
    """
    
    def __init__(self, config=None):
        """
        Initialize the learning TAAT Agent.
        
        Args:
            config: Agent configuration (loads from environment if None)
        """
        # Load learning configuration
        self.learning_config = config or load_learning_config()
        
        # Initialize enhanced agent with learning config
        super().__init__(self.learning_config)
        
        # Initialize learning manager
        self.learning_manager = LearningManager(
            self.learning_config.learning,
            self.db_manager,
            self.memory_manager
        )
        
    async def process_input(self, input_data: Any, input_type: str = "text") -> Dict[str, Any]:
        """
        Process a single input through the perception-cognition-action loop with learning.
        
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
        
        # 3. Enhance context with relevant patterns
        patterns = await self.learning_manager.get_relevant_patterns(processed_input)
        context["learning"] = {
            "relevant_patterns": patterns
        }
        
        # 4. Cognition: Generate response
        response = await self.cognition.process(processed_input, context)
        
        # 5. Action: Execute response
        result = await self.action.execute(response)
        
        # 6. Memory: Update all memory systems
        await self.memory_manager.update_memories(processed_input, response, result)
        
        # 7. Learning: Process outcome
        if "outcome" in result:
            outcome_data = {
                "state": processed_input,
                "action": response,
                "next_state": result,
                "outcome": result.get("outcome"),
                "profit_loss": result.get("profit_loss", 0)
            }
            await self.learning_manager.process_outcome(outcome_data)
        
        return result
    
    async def process_feedback(self, feedback_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process explicit feedback.
        
        Args:
            feedback_data: Feedback data
            
        Returns:
            Processing result
        """
        return await self.learning_manager.process_feedback(feedback_data)
    
    async def get_performance_metrics(self, timeframe: str = "all") -> Dict[str, Any]:
        """
        Get performance metrics.
        
        Args:
            timeframe: Time frame for metrics
            
        Returns:
            Performance metrics
        """
        return await self.learning_manager.get_performance_metrics(timeframe)
    
    async def run_learning_cycle(self) -> Dict[str, Any]:
        """
        Run a complete learning cycle.
        
        Returns:
            Learning cycle results
        """
        return await self.learning_manager.run_learning_cycle()
    
    def start_background_learning(self) -> None:
        """Start background learning."""
        self.learning_manager.start_background_learning()
    
    def stop_background_learning(self) -> None:
        """Stop background learning."""
        self.learning_manager.stop_background_learning()
