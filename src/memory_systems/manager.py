"""
Memory Manager for integrating all memory systems.

This module provides a unified memory manager that coordinates all memory systems,
including working memory, episodic memory, semantic memory, and procedural memory.
"""

import json
from typing import Dict, List, Any, Optional

from src.agent_core.memory.memory import WorkingMemory
from src.memory_systems.config import MemoryConfig
from src.memory_systems.database import DatabaseManager
from src.memory_systems.episodic import EpisodicMemory
from src.memory_systems.semantic import SemanticMemory
from src.memory_systems.procedural import ProceduralMemory


class MemoryManager:
    """
    Memory Manager for integrating all memory systems.
    
    Coordinates working memory, episodic memory, semantic memory, and procedural memory
    to provide a unified interface for memory operations.
    """
    
    def __init__(self, config: MemoryConfig):
        """
        Initialize the memory manager.
        
        Args:
            config: Memory configuration
        """
        self.config = config
        
        # Initialize database manager
        self.db_manager = DatabaseManager(config.database)
        self.db_manager.create_tables()
        
        # Initialize memory systems
        self.working_memory = WorkingMemory(max_history=config.max_episodic_memories)
        self.episodic_memory = EpisodicMemory(config.vector_db, config.embedding)
        self.semantic_memory = SemanticMemory(self.db_manager)
        self.procedural_memory = ProceduralMemory(self.db_manager)
    
    async def _get_embedding_text(self, input_data: Any) -> str:
        """
        Extract text for embedding from input data.
        
        Args:
            input_data: Input data
            
        Returns:
            Text for embedding
        """
        if isinstance(input_data, str):
            return input_data
        
        if isinstance(input_data, dict):
            text_parts = []
            
            # Extract content from input
            if "content" in input_data:
                text_parts.append(input_data["content"])
            
            # Extract trader and symbol information
            if "trader_id" in input_data:
                text_parts.append(f"Trader: {input_data['trader_id']}")
            
            if "symbol" in input_data:
                text_parts.append(f"Symbol: {input_data['symbol']}")
            
            if "action" in input_data:
                text_parts.append(f"Action: {input_data['action']}")
            
            return " ".join(text_parts)
        
        return str(input_data)
    
    async def get_context(self, current_input: Any) -> Dict[str, Any]:
        """
        Get comprehensive context for decision-making.
        
        Args:
            current_input: Current input data
            
        Returns:
            Comprehensive context from all memory systems
        """
        # Get basic context from working memory
        context = self.working_memory.get_context()
        
        # Extract query text for similarity search
        query_text = await self._get_embedding_text(current_input)
        
        # Get similar experiences from episodic memory
        similar_experiences = await self.episodic_memory.retrieve_similar_experiences(
            query_text, limit=self.config.max_episodic_memories
        )
        
        # Extract trader and market information
        trader_info = {}
        market_info = {}
        
        if isinstance(current_input, dict):
            if "trader_id" in current_input:
                trader_info = await self.semantic_memory.get_trader_profile(current_input["trader_id"])
            
            if "symbol" in current_input:
                market_info = await self.semantic_memory.get_market_knowledge(current_input["symbol"])
        
        # Get relevant action patterns
        action_patterns = await self.procedural_memory.get_relevant_patterns(
            current_input if isinstance(current_input, dict) else {},
            limit=self.config.max_procedural_patterns
        )
        
        # Combine all context
        enhanced_context = {
            "working_memory": context,
            "episodic_memory": {
                "similar_experiences": similar_experiences
            },
            "semantic_memory": {
                "trader_info": trader_info,
                "market_info": market_info
            },
            "procedural_memory": {
                "action_patterns": action_patterns
            }
        }
        
        return enhanced_context
    
    async def update_memories(
        self, input_data: Any, response: Any, result: Any, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Update all memory systems with new interaction.
        
        Args:
            input_data: Input data
            response: Response data
            result: Result data
            metadata: Additional metadata
        """
        # Update working memory
        self.working_memory.update(input_data, response, result)
        
        # Prepare experience for episodic memory
        experience = {
            "input": input_data,
            "response": response,
            "result": result
        }
        
        if metadata:
            experience.update(metadata)
        
        # Store in episodic memory
        await self.episodic_memory.store_experience(experience)
        
        # Update semantic memory if applicable
        if isinstance(input_data, dict):
            # Update trader profile
            if "trader_id" in input_data:
                trader_id = input_data["trader_id"]
                
                # Check for outcome information
                if isinstance(result, dict) and "outcome" in result:
                    await self.semantic_memory.update_trader_reliability(
                        trader_id, result["outcome"]
                    )
                
                # Add trade to history if applicable
                if "symbol" in input_data and "action" in input_data:
                    trade_data = {
                        "symbol": input_data["symbol"],
                        "action": input_data["action"],
                        "timestamp": metadata.get("timestamp") if metadata else None,
                        "result": result
                    }
                    await self.semantic_memory.add_trade_to_history(trader_id, trade_data)
            
            # Update market knowledge
            if "symbol" in input_data:
                symbol = input_data["symbol"]
                
                # Add signal to market if applicable
                if "action" in input_data:
                    signal_data = {
                        "trader_id": input_data.get("trader_id"),
                        "action": input_data["action"],
                        "timestamp": metadata.get("timestamp") if metadata else None,
                        "result": result
                    }
                    await self.semantic_memory.add_signal_to_market(symbol, signal_data)
        
        # Update procedural memory if applicable
        if isinstance(response, dict) and "action_sequence" in response:
            # Store action sequence pattern
            pattern_type = "action_sequence"
            await self.procedural_memory.store_pattern(
                pattern_type,
                response["action_sequence"],
                success=result.get("success", False) if isinstance(result, dict) else False
            )
        
        # Store tool usage patterns if applicable
        if isinstance(response, dict) and "tool" in response:
            pattern_type = "tool_usage"
            tool_data = {
                "tool": response["tool"],
                "parameters": response.get("parameters", {})
            }
            await self.procedural_memory.store_pattern(
                pattern_type,
                tool_data,
                success=result.get("success", False) if isinstance(result, dict) else False
            )
    
    async def clear_working_memory(self) -> None:
        """Clear working memory."""
        self.working_memory.clear_history()
        self.working_memory.clear_state()
    
    async def clear_caches(self) -> None:
        """Clear all memory caches."""
        self.semantic_memory.clear_caches()
        self.procedural_memory.clear_cache()
