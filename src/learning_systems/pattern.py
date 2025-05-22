"""
Pattern recognition system for the TAAT AI Agent.

This module provides pattern recognition functionality for detecting patterns
in trading strategies, signals, and outcomes.
"""

import json
import uuid
import datetime
import hashlib
from typing import Dict, List, Any, Optional, Tuple


class PatternRecognition:
    """
    Pattern recognition system for the TAAT AI Agent.
    
    Detects patterns in trading strategies, signals, and outcomes.
    """
    
    def __init__(self, config, db_manager, episodic_memory):
        """
        Initialize the pattern recognition system.
        
        Args:
            config: Pattern recognition configuration
            db_manager: Database manager
            episodic_memory: Episodic memory system
        """
        self.config = config
        self.db_manager = db_manager
        self.episodic_memory = episodic_memory
        self.min_pattern_occurrences = config.min_pattern_occurrences
        self.min_pattern_confidence = config.min_pattern_confidence
        self.max_patterns_per_type = config.max_patterns_per_type
        self.pattern_similarity_threshold = config.pattern_similarity_threshold
        
        # Pattern cache
        self.pattern_cache = {}
    
    async def detect_patterns(self, data_type: str, timeframe: str = "all") -> List[Dict[str, Any]]:
        """
        Detect patterns in historical data.
        
        Args:
            data_type: Type of data to analyze (trades, signals, etc.)
            timeframe: Time frame for analysis
            
        Returns:
            Detected patterns
        """
        # Get historical data
        if data_type == "trades":
            data = await self._get_trade_outcomes(timeframe)
        elif data_type == "signals":
            data = await self._get_trade_signals(timeframe)
        elif data_type == "feedback":
            data = await self._get_feedback_history(timeframe)
        else:
            raise ValueError(f"Unsupported data type: {data_type}")
        
        # Detect patterns based on data type
        if data_type == "trades":
            return await self._detect_trade_patterns(data)
        elif data_type == "signals":
            return await self._detect_signal_patterns(data)
        elif data_type == "feedback":
            return await self._detect_feedback_patterns(data)
        
        return []
    
    async def _detect_trade_patterns(self, trades: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Detect patterns in trade data.
        
        Args:
            trades: Trade data
            
        Returns:
            Detected patterns
        """
        patterns = []
        
        # Group trades by symbol
        trades_by_symbol = {}
        for trade in trades:
            symbol = trade.get("symbol")
            if not symbol:
                continue
            if symbol not in trades_by_symbol:
                trades_by_symbol[symbol] = []
            trades_by_symbol[symbol].append(trade)
        
        # Analyze each symbol
        for symbol, symbol_trades in trades_by_symbol.items():
            # Calculate success rate
            total = len(symbol_trades)
            if total < self.min_pattern_occurrences:
                continue
            
            successful = sum(1 for t in symbol_trades if t.get("outcome") == "success")
            success_rate = successful / total if total > 0 else 0
            
            # Check if this is a high-success pattern
            if success_rate >= self.min_pattern_confidence:
                pattern = {
                    "id": str(uuid.uuid4()),
                    "type": "high_success_symbol",
                    "symbol": symbol,
                    "success_rate": success_rate,
                    "sample_size": total,
                    "confidence": min(1.0, total / (self.min_pattern_occurrences * 2)) * success_rate,
                    "detected_at": datetime.datetime.utcnow().isoformat()
                }
                patterns.append(pattern)
                
                # Store pattern
                await self._store_pattern(pattern)
        
        # Group trades by trader
        trades_by_trader = {}
        for trade in trades:
            trader_id = trade.get("trader_id")
            if not trader_id:
                continue
            if trader_id not in trades_by_trader:
                trades_by_trader[trader_id] = []
            trades_by_trader[trader_id].append(trade)
        
        # Analyze each trader
        for trader_id, trader_trades in trades_by_trader.items():
            # Calculate success rate
            total = len(trader_trades)
            if total < self.min_pattern_occurrences:
                continue
            
            successful = sum(1 for t in trader_trades if t.get("outcome") == "success")
            success_rate = successful / total if total > 0 else 0
            
            # Check if this is a reliable trader pattern
            if success_rate >= self.min_pattern_confidence:
                pattern = {
                    "id": str(uuid.uuid4()),
                    "type": "reliable_trader",
                    "trader_id": trader_id,
                    "success_rate": success_rate,
                    "sample_size": total,
                    "confidence": min(1.0, total / (self.min_pattern_occurrences * 2)) * success_rate,
                    "detected_at": datetime.datetime.utcnow().isoformat()
                }
                patterns.append(pattern)
                
                # Store pattern
                await self._store_pattern(pattern)
        
        # Analyze action-symbol combinations
        action_symbol_patterns = await self._detect_action_symbol_patterns(trades)
        patterns.extend(action_symbol_patterns)
        
        # Limit number of patterns
        patterns.sort(key=lambda p: p.get("confidence", 0), reverse=True)
        return patterns[:self.max_patterns_per_type]
    
    async def _detect_action_symbol_patterns(self, trades: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Detect patterns in action-symbol combinations.
        
        Args:
            trades: Trade data
            
        Returns:
            Detected patterns
        """
        patterns = []
        
        # Group trades by action-symbol combination
        action_symbol_trades = {}
        for trade in trades:
            action = trade.get("action")
            symbol = trade.get("symbol")
            if not action or not symbol:
                continue
            
            key = f"{action}:{symbol}"
            if key not in action_symbol_trades:
                action_symbol_trades[key] = []
            action_symbol_trades[key].append(trade)
        
        # Analyze each action-symbol combination
        for key, combo_trades in action_symbol_trades.items():
            # Calculate success rate
            total = len(combo_trades)
            if total < self.min_pattern_occurrences:
                continue
            
            successful = sum(1 for t in combo_trades if t.get("outcome") == "success")
            success_rate = successful / total if total > 0 else 0
            
            # Check if this is a successful action-symbol pattern
            if success_rate >= self.min_pattern_confidence:
                action, symbol = key.split(":")
                pattern = {
                    "id": str(uuid.uuid4()),
                    "type": "action_symbol_success",
                    "action": action,
                    "symbol": symbol,
                    "success_rate": success_rate,
                    "sample_size": total,
                    "confidence": min(1.0, total / (self.min_pattern_occurrences * 2)) * success_rate,
                    "detected_at": datetime.datetime.utcnow().isoformat()
                }
                patterns.append(pattern)
                
                # Store pattern
                await self._store_pattern(pattern)
        
        return patterns
    
    async def _detect_signal_patterns(self, signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Detect patterns in signal data.
        
        Args:
            signals: Signal data
            
        Returns:
            Detected patterns
        """
        patterns = []
        
        # Group signals by trader
        signals_by_trader = {}
        for signal in signals:
            trader_id = signal.get("trader_id")
            if not trader_id:
                continue
            if trader_id not in signals_by_trader:
                signals_by_trader[trader_id] = []
            signals_by_trader[trader_id].append(signal)
        
        # Analyze each trader's signal frequency
        for trader_id, trader_signals in signals_by_trader.items():
            # Calculate signal frequency
            if len(trader_signals) < self.min_pattern_occurrences:
                continue
            
            # Sort signals by timestamp
            trader_signals.sort(key=lambda s: s.get("timestamp", ""))
            
            # Calculate average time between signals
            time_diffs = []
            for i in range(1, len(trader_signals)):
                try:
                    time1 = datetime.datetime.fromisoformat(trader_signals[i-1].get("timestamp", ""))
                    time2 = datetime.datetime.fromisoformat(trader_signals[i].get("timestamp", ""))
                    diff_seconds = (time2 - time1).total_seconds()
                    time_diffs.append(diff_seconds)
                except (ValueError, TypeError):
                    continue
            
            if not time_diffs:
                continue
            
            avg_time_between_signals = sum(time_diffs) / len(time_diffs)
            
            # Check if this is a frequent signal pattern
            if avg_time_between_signals <= 86400:  # 24 hours
                pattern = {
                    "id": str(uuid.uuid4()),
                    "type": "frequent_signaler",
                    "trader_id": trader_id,
                    "avg_time_between_signals": avg_time_between_signals,
                    "signals_per_day": 86400 / avg_time_between_signals if avg_time_between_signals > 0 else 0,
                    "sample_size": len(trader_signals),
                    "confidence": min(1.0, len(trader_signals) / (self.min_pattern_occurrences * 2)),
                    "detected_at": datetime.datetime.utcnow().isoformat()
                }
                patterns.append(pattern)
                
                # Store pattern
                await self._store_pattern(pattern)
        
        # Analyze signal content patterns
        content_patterns = await self._detect_signal_content_patterns(signals)
        patterns.extend(content_patterns)
        
        # Limit number of patterns
        patterns.sort(key=lambda p: p.get("confidence", 0), reverse=True)
        return patterns[:self.max_patterns_per_type]
    
    async def _detect_signal_content_patterns(self, signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Detect patterns in signal content.
        
        Args:
            signals: Signal data
            
        Returns:
            Detected patterns
        """
        patterns = []
        
        # Extract signal content
        signal_contents = []
        for signal in signals:
            content = signal.get("content", "")
            if not content:
                continue
            
            signal_contents.append({
                "content": content,
                "trader_id": signal.get("trader_id"),
                "timestamp": signal.get("timestamp")
            })
        
        # Use episodic memory to find similar content
        if signal_contents and self.episodic_memory:
            try:
                # Get embeddings for signal contents
                for content_data in signal_contents:
                    similar_experiences = await self.episodic_memory.retrieve_similar_experiences(
                        content_data["content"], limit=5
                    )
                    
                    if similar_experiences:
                        # Check if there's a pattern of similar content
                        if len(similar_experiences) >= self.min_pattern_occurrences:
                            pattern = {
                                "id": str(uuid.uuid4()),
                                "type": "similar_content",
                                "content_sample": content_data["content"][:100] + "..." if len(content_data["content"]) > 100 else content_data["content"],
                                "trader_id": content_data["trader_id"],
                                "similar_count": len(similar_experiences),
                                "avg_similarity": sum(exp.get("similarity_score", 0) for exp in similar_experiences) / len(similar_experiences),
                                "confidence": min(1.0, len(similar_experiences) / (self.min_pattern_occurrences * 2)),
                                "detected_at": datetime.datetime.utcnow().isoformat()
                            }
                            patterns.append(pattern)
                            
                            # Store pattern
                            await self._store_pattern(pattern)
            except Exception as e:
                # Log error
                print(f"Error detecting signal content patterns: {e}")
        
        return patterns
    
    async def _detect_feedback_patterns(self, feedback: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Detect patterns in feedback data.
        
        Args:
            feedback: Feedback data
            
        Returns:
            Detected patterns
        """
        patterns = []
        
        # Group feedback by type
        feedback_by_type = {}
        for fb in feedback:
            fb_type = fb.get("type", "general")
            if fb_type not in feedback_by_type:
                feedback_by_type[fb_type] = []
            feedback_by_type[fb_type].append(fb)
        
        # Analyze each feedback type
        for fb_type, type_feedback in feedback_by_type.items():
            # Calculate average value
            if len(type_feedback) < self.min_pattern_occurrences:
                continue
            
            values = [fb.get("value", 0) for fb in type_feedback if "value" in fb]
            if not values:
                continue
            
            avg_value = sum(values) / len(values)
            
            # Check if this is a consistent feedback pattern
            if abs(avg_value) >= self.min_pattern_confidence:
                pattern = {
                    "id": str(uuid.uuid4()),
                    "type": "consistent_feedback",
                    "feedback_type": fb_type,
                    "avg_value": avg_value,
                    "sample_size": len(values),
                    "confidence": min(1.0, len(values) / (self.min_pattern_occurrences * 2)) * abs(avg_value),
                    "detected_at": datetime.datetime.utcnow().isoformat()
                }
                patterns.append(pattern)
                
                # Store pattern
                await self._store_pattern(pattern)
        
        # Limit number of patterns
        patterns.sort(key=lambda p: p.get("confidence", 0), reverse=True)
        return patterns[:self.max_patterns_per_type]
    
    async def get_patterns_by_type(self, pattern_type: str, min_confidence: float = 0.0) -> List[Dict[str, Any]]:
        """
        Get patterns by type.
        
        Args:
            pattern_type: Pattern type
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of patterns
        """
        try:
            patterns = await self.db_manager.get_patterns_by_type(pattern_type)
            
            # Filter by confidence
            if min_confidence > 0:
                patterns = [p for p in patterns if p.get("confidence", 0) >= min_confidence]
            
            # Sort by confidence
            patterns.sort(key=lambda p: p.get("confidence", 0), reverse=True)
            
            return patterns[:self.max_patterns_per_type]
        except Exception as e:
            # Log error and return empty list
            print(f"Error getting patterns by type: {e}")
            return []
    
    async def get_relevant_patterns(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Get patterns relevant to the current context.
        
        Args:
            context: Current context
            
        Returns:
            List of relevant patterns
        """
        relevant_patterns = []
        
        # Extract context information
        trader_id = context.get("trader_id")
        symbol = context.get("symbol")
        action = context.get("action")
        
        # Get trader-specific patterns
        if trader_id:
            trader_patterns = await self.get_patterns_by_type("reliable_trader", min_confidence=self.min_pattern_confidence)
            trader_patterns.extend(await self.get_patterns_by_type("frequent_signaler", min_confidence=self.min_pattern_confidence))
            
            # Filter for this trader
            trader_patterns = [p for p in trader_patterns if p.get("trader_id") == trader_id]
            relevant_patterns.extend(trader_patterns)
        
        # Get symbol-specific patterns
        if symbol:
            symbol_patterns = await self.get_patterns_by_type("high_success_symbol", min_confidence=self.min_pattern_confidence)
            
            # Filter for this symbol
            symbol_patterns = [p for p in symbol_patterns if p.get("symbol") == symbol]
            relevant_patterns.extend(symbol_patterns)
        
        # Get action-symbol patterns
        if action and symbol:
            action_symbol_patterns = await self.get_patterns_by_type("action_symbol_success", min_confidence=self.min_pattern_confidence)
            
            # Filter for this action and symbol
            action_symbol_patterns = [
                p for p in action_symbol_patterns 
                if p.get("action") == action and p.get("symbol") == symbol
            ]
            relevant_patterns.extend(action_symbol_patterns)
        
        # Sort by confidence and limit
        relevant_patterns.sort(key=lambda p: p.get("confidence", 0), reverse=True)
        return relevant_patterns[:self.max_patterns_per_type]
    
    async def _get_trade_outcomes(self, timeframe: str) -> List[Dict[str, Any]]:
        """
        Get trade outcomes for the specified timeframe.
        
        Args:
            timeframe: Time frame for outcomes
            
        Returns:
            List of trade outcomes
        """
        try:
            return await self.db_manager.get_trade_outcomes(timeframe)
        except Exception as e:
            # Log error and return empty list
            print(f"Error getting trade outcomes: {e}")
            return []
    
    async def _get_trade_signals(self, timeframe: str) -> List[Dict[str, Any]]:
        """
        Get trade signals for the specified timeframe.
        
        Args:
            timeframe: Time frame for signals
            
        Returns:
            List of trade signals
        """
        try:
            return await self.db_manager.get_trade_signals(timeframe)
        except Exception as e:
            # Log error and return empty list
            print(f"Error getting trade signals: {e}")
            return []
    
    async def _get_feedback_history(self, timeframe: str) -> List[Dict[str, Any]]:
        """
        Get feedback history for the specified timeframe.
        
        Args:
            timeframe: Time frame for feedback
            
        Returns:
            List of feedback items
        """
        try:
            return await self.db_manager.get_feedback_history(timeframe)
        except Exception as e:
            # Log error and return empty list
            print(f"Error getting feedback history: {e}")
            return []
    
    async def _store_pattern(self, pattern: Dict[str, Any]) -> None:
        """
        Store pattern in database.
        
        Args:
            pattern: Pattern to store
        """
        try:
            # Generate pattern key
            pattern_type = pattern.get("type", "unknown")
            pattern_str = json.dumps(pattern, sort_keys=True)
            pattern_hash = hashlib.md5(pattern_str.encode()).hexdigest()
            pattern_key = f"{pattern_type}:{pattern_hash}"
            
            # Check cache to avoid duplicates
            if pattern_key in self.pattern_cache:
                return
            
            # Store in database
            await self.db_manager.store_pattern(pattern_type, pattern_key, pattern)
            
            # Update cache
            self.pattern_cache[pattern_key] = pattern
        except Exception as e:
            # Log error
            print(f"Error storing pattern: {e}")
    
    def clear_cache(self) -> None:
        """Clear pattern cache."""
        self.pattern_cache = {}
