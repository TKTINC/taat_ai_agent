"""
Procedural memory implementation for action patterns and learned behaviors.

This module provides procedural memory functionality for storing and retrieving
patterns of successful actions and learned behaviors.
"""

import json
import hashlib
from typing import Dict, List, Any, Optional, Tuple

from src.memory_systems.database import DatabaseManager


class ProceduralMemory:
    """
    Procedural memory implementation for action patterns and learned behaviors.
    
    Stores and retrieves patterns of successful actions and learned behaviors.
    """
    
    def __init__(self, db_manager: DatabaseManager):
        """
        Initialize procedural memory.
        
        Args:
            db_manager: Database manager
        """
        self.db_manager = db_manager
        self.pattern_cache = {}
    
    def _generate_pattern_key(self, pattern_data: Any) -> str:
        """
        Generate a unique key for a pattern.
        
        Args:
            pattern_data: Pattern data
            
        Returns:
            Pattern key
        """
        # Convert to JSON string and hash
        pattern_str = json.dumps(pattern_data, sort_keys=True)
        return hashlib.md5(pattern_str.encode()).hexdigest()
    
    async def store_pattern(
        self, pattern_type: str, pattern_data: Any, success: bool = True
    ) -> Dict[str, Any]:
        """
        Store an action pattern.
        
        Args:
            pattern_type: Type of pattern (e.g., "trade_strategy", "tool_sequence")
            pattern_data: Pattern data
            success: Whether the pattern was successful
            
        Returns:
            Stored pattern
        """
        # Generate pattern key
        pattern_key = self._generate_pattern_key(pattern_data)
        
        # Check if pattern exists
        existing_pattern = await self.db_manager.get_action_pattern(pattern_type, pattern_key)
        
        if existing_pattern:
            # Update existing pattern
            updates = {
                "success_count": existing_pattern["success_count"] + (1 if success else 0),
                "failure_count": existing_pattern["failure_count"] + (0 if success else 1),
            }
            
            # Calculate effectiveness
            total = updates["success_count"] + updates["failure_count"]
            if total > 0:
                updates["effectiveness"] = updates["success_count"] / total
            
            # Update pattern data
            pattern_data_dict = existing_pattern.get("data", {}).get("pattern_data", {})
            updates["data"] = {
                "pattern_data": pattern_data_dict
            }
            
            # Update pattern
            updated_pattern = await self.db_manager.update_action_pattern(
                pattern_type, pattern_key, updates
            )
            
            # Update cache
            if updated_pattern:
                cache_key = f"{pattern_type}:{pattern_key}"
                self.pattern_cache[cache_key] = updated_pattern
            
            return updated_pattern or existing_pattern
        else:
            # Create new pattern
            new_pattern = await self.db_manager.create_action_pattern(
                pattern_type, 
                pattern_key, 
                {
                    "success_count": 1 if success else 0,
                    "failure_count": 0 if success else 1,
                    "effectiveness": 1.0 if success else 0.0,
                    "data": {
                        "pattern_data": pattern_data
                    }
                }
            )
            
            # Update cache
            cache_key = f"{pattern_type}:{pattern_key}"
            self.pattern_cache[cache_key] = new_pattern
            
            return new_pattern
    
    async def get_pattern(self, pattern_type: str, pattern_key: str) -> Optional[Dict[str, Any]]:
        """
        Get an action pattern.
        
        Args:
            pattern_type: Pattern type
            pattern_key: Pattern key
            
        Returns:
            Action pattern or None if not found
        """
        # Check cache first
        cache_key = f"{pattern_type}:{pattern_key}"
        if cache_key in self.pattern_cache:
            return self.pattern_cache[cache_key]
        
        # Get from database
        pattern = await self.db_manager.get_action_pattern(pattern_type, pattern_key)
        
        # Update cache
        if pattern:
            self.pattern_cache[cache_key] = pattern
        
        return pattern
    
    async def get_patterns_by_type(
        self, pattern_type: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get action patterns by type.
        
        Args:
            pattern_type: Pattern type
            limit: Maximum number of patterns to return
            
        Returns:
            List of action patterns
        """
        return await self.db_manager.get_action_patterns_by_type(pattern_type, limit)
    
    async def get_most_effective_patterns(
        self, pattern_type: str, min_effectiveness: float = 0.5, limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get most effective action patterns.
        
        Args:
            pattern_type: Pattern type
            min_effectiveness: Minimum effectiveness threshold
            limit: Maximum number of patterns to return
            
        Returns:
            List of action patterns
        """
        patterns = await self.get_patterns_by_type(pattern_type, limit=limit*2)
        
        # Filter by effectiveness
        effective_patterns = [
            p for p in patterns 
            if p.get("effectiveness", 0) >= min_effectiveness
        ]
        
        # Sort by effectiveness
        effective_patterns.sort(key=lambda p: p.get("effectiveness", 0), reverse=True)
        
        return effective_patterns[:limit]
    
    async def get_relevant_patterns(
        self, context: Dict[str, Any], limit: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Get patterns relevant to the current context.
        
        Args:
            context: Current context
            limit: Maximum number of patterns to return
            
        Returns:
            List of relevant patterns
        """
        relevant_patterns = []
        
        # Extract context information
        pattern_types = []
        
        if "trader_id" in context:
            pattern_types.append(f"trader:{context['trader_id']}")
        
        if "symbol" in context:
            pattern_types.append(f"symbol:{context['symbol']}")
        
        if "action" in context:
            pattern_types.append(f"action:{context['action']}")
        
        # Always include general patterns
        pattern_types.append("general")
        
        # Get patterns for each type
        for pattern_type in pattern_types:
            patterns = await self.get_most_effective_patterns(
                pattern_type, min_effectiveness=0.6, limit=limit
            )
            relevant_patterns.extend(patterns)
        
        # Sort by effectiveness and limit
        relevant_patterns.sort(key=lambda p: p.get("effectiveness", 0), reverse=True)
        
        return relevant_patterns[:limit]
    
    def clear_cache(self) -> None:
        """Clear pattern cache."""
        self.pattern_cache = {}
