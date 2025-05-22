"""
Semantic memory implementation for trader profiles and market knowledge.

This module provides semantic memory functionality for storing and retrieving
structured knowledge about traders and markets.
"""

import json
from typing import Dict, List, Any, Optional

from src.memory_systems.database import DatabaseManager


class SemanticMemory:
    """
    Semantic memory implementation for trader profiles and market knowledge.
    
    Stores and retrieves structured knowledge about traders and markets.
    """
    
    def __init__(self, db_manager: DatabaseManager):
        """
        Initialize semantic memory.
        
        Args:
            db_manager: Database manager
        """
        self.db_manager = db_manager
        self.trader_cache = {}
        self.market_cache = {}
    
    async def get_trader_profile(self, trader_id: str) -> Dict[str, Any]:
        """
        Get a trader's profile from semantic memory.
        
        Args:
            trader_id: Trader ID
            
        Returns:
            Trader profile
        """
        # Check cache first
        if trader_id in self.trader_cache:
            return self.trader_cache[trader_id]
        
        # Try to get from database
        profile = await self.db_manager.get_trader_profile(trader_id)
        
        # If not found, create new profile
        if not profile:
            profile = await self.db_manager.create_trader_profile(trader_id, {
                "successful_trades": 0,
                "failed_trades": 0,
                "reliability": 0.5,
                "data": {
                    "trade_history": [],
                    "preferences": {},
                    "notes": ""
                }
            })
        
        # Update cache
        self.trader_cache[trader_id] = profile
        
        return profile
    
    async def update_trader_profile(self, trader_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update a trader's profile in semantic memory.
        
        Args:
            trader_id: Trader ID
            updates: Profile updates
            
        Returns:
            Updated trader profile
        """
        # Get current profile
        profile = await self.get_trader_profile(trader_id)
        
        # Apply updates
        updated_profile = await self.db_manager.update_trader_profile(trader_id, updates)
        
        # Update cache
        if updated_profile:
            self.trader_cache[trader_id] = updated_profile
        
        return updated_profile or profile
    
    async def update_trader_reliability(self, trader_id: str, outcome: str) -> Dict[str, Any]:
        """
        Update a trader's reliability based on trade outcome.
        
        Args:
            trader_id: Trader ID
            outcome: Trade outcome ("success" or "failure")
            
        Returns:
            Updated trader profile
        """
        # Get current profile
        profile = await self.get_trader_profile(trader_id)
        
        # Update reliability metrics
        updates = {}
        if outcome == "success":
            updates["successful_trades"] = profile.get("successful_trades", 0) + 1
        elif outcome == "failure":
            updates["failed_trades"] = profile.get("failed_trades", 0) + 1
        
        # Calculate new reliability
        total_trades = updates.get("successful_trades", profile.get("successful_trades", 0)) + \
                      updates.get("failed_trades", profile.get("failed_trades", 0))
        
        if total_trades > 0:
            updates["reliability"] = updates.get("successful_trades", profile.get("successful_trades", 0)) / total_trades
        
        # Update profile
        return await self.update_trader_profile(trader_id, updates)
    
    async def add_trade_to_history(self, trader_id: str, trade_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add a trade to a trader's history.
        
        Args:
            trader_id: Trader ID
            trade_data: Trade data
            
        Returns:
            Updated trader profile
        """
        # Get current profile
        profile = await self.get_trader_profile(trader_id)
        
        # Get current trade history
        profile_data = profile.get("data", {})
        trade_history = profile_data.get("trade_history", [])
        
        # Add new trade
        trade_history.append(trade_data)
        
        # Update profile
        updates = {
            "data": {
                "trade_history": trade_history
            }
        }
        
        return await self.update_trader_profile(trader_id, updates)
    
    async def get_market_knowledge(self, symbol: str) -> Dict[str, Any]:
        """
        Get market knowledge from semantic memory.
        
        Args:
            symbol: Market symbol
            
        Returns:
            Market knowledge
        """
        # Check cache first
        if symbol in self.market_cache:
            return self.market_cache[symbol]
        
        # Try to get from database
        knowledge = await self.db_manager.get_market_knowledge(symbol)
        
        # If not found, create new knowledge
        if not knowledge:
            knowledge = await self.db_manager.create_market_knowledge(symbol, {
                "name": symbol,
                "sector": "Unknown",
                "data": {
                    "price_history": [],
                    "trade_signals": [],
                    "notes": ""
                }
            })
        
        # Update cache
        self.market_cache[symbol] = knowledge
        
        return knowledge
    
    async def update_market_knowledge(self, symbol: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update market knowledge in semantic memory.
        
        Args:
            symbol: Market symbol
            updates: Knowledge updates
            
        Returns:
            Updated market knowledge
        """
        # Get current knowledge
        knowledge = await self.get_market_knowledge(symbol)
        
        # Apply updates
        updated_knowledge = await self.db_manager.update_market_knowledge(symbol, updates)
        
        # Update cache
        if updated_knowledge:
            self.market_cache[symbol] = updated_knowledge
        
        return updated_knowledge or knowledge
    
    async def add_signal_to_market(self, symbol: str, signal_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add a trade signal to market knowledge.
        
        Args:
            symbol: Market symbol
            signal_data: Signal data
            
        Returns:
            Updated market knowledge
        """
        # Get current knowledge
        knowledge = await self.get_market_knowledge(symbol)
        
        # Get current signals
        knowledge_data = knowledge.get("data", {})
        trade_signals = knowledge_data.get("trade_signals", [])
        
        # Add new signal
        trade_signals.append(signal_data)
        
        # Update knowledge
        updates = {
            "data": {
                "trade_signals": trade_signals
            }
        }
        
        return await self.update_market_knowledge(symbol, updates)
    
    def clear_caches(self) -> None:
        """Clear all caches."""
        self.trader_cache = {}
        self.market_cache = {}
