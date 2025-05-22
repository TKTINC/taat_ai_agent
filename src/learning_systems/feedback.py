"""
Feedback processing system for the TAAT AI Agent.

This module provides feedback processing functionality for integrating explicit
user feedback and implicit feedback from trade outcomes.
"""

import json
import uuid
import datetime
from typing import Dict, List, Any, Optional, Tuple


class FeedbackProcessor:
    """
    Feedback processing system for the TAAT AI Agent.
    
    Processes explicit user feedback and implicit feedback from trade outcomes.
    """
    
    def __init__(self, config, db_manager):
        """
        Initialize the feedback processor.
        
        Args:
            config: Feedback configuration
            db_manager: Database manager
        """
        self.config = config
        self.db_manager = db_manager
        self.positive_threshold = config.positive_threshold
        self.negative_threshold = config.negative_threshold
        self.feedback_weight = config.feedback_weight
        self.outcome_weight = config.outcome_weight
        self.feedback_decay = config.feedback_decay
        
        # Feedback history
        self.feedback_history = []
    
    async def process_user_feedback(self, feedback_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process explicit user feedback.
        
        Args:
            feedback_data: Feedback data
            
        Returns:
            Processed feedback
        """
        # Extract feedback information
        feedback_type = feedback_data.get("type", "general")
        feedback_value = feedback_data.get("value", 0.0)
        feedback_text = feedback_data.get("text", "")
        
        # Normalize feedback value to [-1, 1]
        if isinstance(feedback_value, str):
            if feedback_value.lower() in ["positive", "good", "yes"]:
                normalized_value = 1.0
            elif feedback_value.lower() in ["negative", "bad", "no"]:
                normalized_value = -1.0
            else:
                normalized_value = 0.0
        else:
            # Assume numeric value in range [-1, 1]
            normalized_value = max(-1.0, min(1.0, float(feedback_value)))
        
        # Create feedback record
        feedback_record = {
            "id": str(uuid.uuid4()),
            "type": feedback_type,
            "value": normalized_value,
            "text": feedback_text,
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "source": "user"
        }
        
        # Add to history
        self.feedback_history.append(feedback_record)
        
        # Store feedback in database
        feedback_id = await self._store_feedback(feedback_record)
        
        # Update relevant models based on feedback type
        if feedback_type == "trade_signal":
            await self._update_signal_model(feedback_data, normalized_value)
        elif feedback_type == "trader_reliability":
            await self._update_trader_model(feedback_data, normalized_value)
        elif feedback_type == "strategy":
            await self._update_strategy_model(feedback_data, normalized_value)
        
        return {
            "feedback_id": feedback_id,
            "status": "processed",
            "applied_to": feedback_type,
            "normalized_value": normalized_value
        }
    
    async def process_trade_outcome(self, trade_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process trade outcome as implicit feedback.
        
        Args:
            trade_data: Trade data
            
        Returns:
            Processed outcome
        """
        # Extract trade information
        trade_id = trade_data.get("trade_id", str(uuid.uuid4()))
        symbol = trade_data.get("symbol", "")
        action = trade_data.get("action", "")
        outcome = trade_data.get("outcome", "unknown")
        profit_loss = trade_data.get("profit_loss", 0.0)
        
        # Calculate reward based on outcome
        reward = self._calculate_reward(outcome, profit_loss)
        
        # Create outcome record
        outcome_record = {
            "id": str(uuid.uuid4()),
            "trade_id": trade_id,
            "symbol": symbol,
            "action": action,
            "outcome": outcome,
            "profit_loss": profit_loss,
            "reward": reward,
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "source": "system"
        }
        
        # Add to history
        self.feedback_history.append(outcome_record)
        
        # Store outcome in database
        outcome_id = await self._store_trade_outcome(outcome_record)
        
        # Update models based on outcome
        await self._update_signal_model({
            "symbol": symbol,
            "action": action,
            "outcome": outcome
        }, reward)
        
        if "trader_id" in trade_data:
            await self._update_trader_model({
                "trader_id": trade_data["trader_id"],
                "outcome": outcome
            }, reward)
        
        return {
            "outcome_id": outcome_id,
            "reward": reward,
            "status": "processed"
        }
    
    def _calculate_reward(self, outcome: str, profit_loss: float) -> float:
        """
        Calculate reward based on outcome and profit/loss.
        
        Args:
            outcome: Trade outcome
            profit_loss: Profit or loss amount
            
        Returns:
            Calculated reward
        """
        if outcome.lower() == "success":
            # Positive reward proportional to profit
            return min(1.0, max(0.1, profit_loss / 100))
        elif outcome.lower() == "failure":
            # Negative reward proportional to loss
            return max(-1.0, min(-0.1, profit_loss / 100))
        else:
            # Neutral reward for unknown outcome
            return 0.0
    
    async def _store_feedback(self, feedback_record: Dict[str, Any]) -> str:
        """
        Store feedback in database.
        
        Args:
            feedback_record: Feedback record
            
        Returns:
            Feedback ID
        """
        try:
            # Store in database
            return await self.db_manager.store_feedback(feedback_record)
        except Exception as e:
            # Log error and return ID anyway
            print(f"Error storing feedback: {e}")
            return feedback_record["id"]
    
    async def _store_trade_outcome(self, outcome_record: Dict[str, Any]) -> str:
        """
        Store trade outcome in database.
        
        Args:
            outcome_record: Outcome record
            
        Returns:
            Outcome ID
        """
        try:
            # Store in database
            return await self.db_manager.store_trade_outcome(outcome_record)
        except Exception as e:
            # Log error and return ID anyway
            print(f"Error storing trade outcome: {e}")
            return outcome_record["id"]
    
    async def _update_signal_model(self, data: Dict[str, Any], value: float) -> None:
        """
        Update signal model based on feedback.
        
        Args:
            data: Signal data
            value: Feedback value
        """
        if "symbol" not in data:
            return
        
        symbol = data["symbol"]
        
        try:
            # Get current market knowledge
            market_knowledge = await self.db_manager.get_market_knowledge(symbol)
            
            if not market_knowledge:
                # Create new market knowledge if not exists
                market_knowledge = await self.db_manager.create_market_knowledge(symbol, {})
            
            # Update reliability based on feedback
            market_data = market_knowledge.get("data", {})
            reliability = market_data.get("reliability", 0.5)
            
            # Apply feedback with weight and decay
            new_reliability = (reliability * self.feedback_decay) + (value * (1 - self.feedback_decay))
            
            # Ensure in range [0, 1]
            new_reliability = max(0.0, min(1.0, new_reliability))
            
            # Update market knowledge
            market_data["reliability"] = new_reliability
            
            # Add feedback to history
            if "feedback_history" not in market_data:
                market_data["feedback_history"] = []
            
            market_data["feedback_history"].append({
                "value": value,
                "timestamp": datetime.datetime.utcnow().isoformat(),
                "source": data.get("source", "unknown")
            })
            
            # Limit history size
            if len(market_data["feedback_history"]) > 100:
                market_data["feedback_history"] = market_data["feedback_history"][-100:]
            
            # Update in database
            await self.db_manager.update_market_knowledge(symbol, {"data": market_data})
            
        except Exception as e:
            # Log error
            print(f"Error updating signal model: {e}")
    
    async def _update_trader_model(self, data: Dict[str, Any], value: float) -> None:
        """
        Update trader model based on feedback.
        
        Args:
            data: Trader data
            value: Feedback value
        """
        if "trader_id" not in data:
            return
        
        trader_id = data["trader_id"]
        
        try:
            # Get current trader profile
            trader_profile = await self.db_manager.get_trader_profile(trader_id)
            
            if not trader_profile:
                # Create new trader profile if not exists
                trader_profile = await self.db_manager.create_trader_profile(trader_id, {})
            
            # Update reliability based on feedback
            profile_data = trader_profile.get("data", {})
            reliability = profile_data.get("reliability", 0.5)
            
            # Apply feedback with weight and decay
            new_reliability = (reliability * self.feedback_decay) + (value * (1 - self.feedback_decay))
            
            # Ensure in range [0, 1]
            new_reliability = max(0.0, min(1.0, new_reliability))
            
            # Update trader profile
            profile_data["reliability"] = new_reliability
            
            # Add feedback to history
            if "feedback_history" not in profile_data:
                profile_data["feedback_history"] = []
            
            profile_data["feedback_history"].append({
                "value": value,
                "timestamp": datetime.datetime.utcnow().isoformat(),
                "source": data.get("source", "unknown")
            })
            
            # Limit history size
            if len(profile_data["feedback_history"]) > 100:
                profile_data["feedback_history"] = profile_data["feedback_history"][-100:]
            
            # Update in database
            await self.db_manager.update_trader_profile(trader_id, {"data": profile_data})
            
        except Exception as e:
            # Log error
            print(f"Error updating trader model: {e}")
    
    async def _update_strategy_model(self, data: Dict[str, Any], value: float) -> None:
        """
        Update strategy model based on feedback.
        
        Args:
            data: Strategy data
            value: Feedback value
        """
        if "strategy_id" not in data and "pattern_type" not in data:
            return
        
        strategy_id = data.get("strategy_id")
        pattern_type = data.get("pattern_type")
        pattern_key = data.get("pattern_key")
        
        if not strategy_id and not (pattern_type and pattern_key):
            return
        
        try:
            # Update action pattern if pattern_type and pattern_key are provided
            if pattern_type and pattern_key:
                # Get current action pattern
                action_pattern = await self.db_manager.get_action_pattern(pattern_type, pattern_key)
                
                if action_pattern:
                    # Update effectiveness based on feedback
                    effectiveness = action_pattern.get("effectiveness", 0.5)
                    
                    # Apply feedback with weight and decay
                    new_effectiveness = (effectiveness * self.feedback_decay) + (value * (1 - self.feedback_decay))
                    
                    # Ensure in range [0, 1]
                    new_effectiveness = max(0.0, min(1.0, new_effectiveness))
                    
                    # Update success/failure counts
                    success_count = action_pattern.get("success_count", 0)
                    failure_count = action_pattern.get("failure_count", 0)
                    
                    if value > self.positive_threshold:
                        success_count += 1
                    elif value < self.negative_threshold:
                        failure_count += 1
                    
                    # Update action pattern
                    await self.db_manager.update_action_pattern(
                        pattern_type,
                        pattern_key,
                        {
                            "effectiveness": new_effectiveness,
                            "success_count": success_count,
                            "failure_count": failure_count
                        }
                    )
            
        except Exception as e:
            # Log error
            print(f"Error updating strategy model: {e}")
    
    async def get_recent_feedback(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent feedback.
        
        Args:
            limit: Maximum number of feedback items to return
            
        Returns:
            List of recent feedback items
        """
        # Sort by timestamp (descending)
        sorted_feedback = sorted(
            self.feedback_history,
            key=lambda x: x.get("timestamp", ""),
            reverse=True
        )
        
        return sorted_feedback[:limit]
