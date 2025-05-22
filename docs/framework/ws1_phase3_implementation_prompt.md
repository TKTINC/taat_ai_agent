# Agent Foundation - Phase 3: Learning Systems Implementation Prompt

## Objective
Implement learning systems for the TAAT AI Agent, including reinforcement learning mechanisms, feedback processing, performance tracking and metrics, and pattern recognition for successful strategies.

## Context
In Phase 1, we established the core agent architecture with the perception-cognition-action loop. In Phase 2, we implemented advanced memory systems including episodic, semantic, and procedural memory. Phase 3 builds upon these foundations to create learning mechanisms that enable the agent to improve over time based on experience and feedback.

## Requirements

1. **Reinforcement Learning Mechanisms**
   - Implement a reward system for successful trade signal interpretations
   - Create a policy learning framework for action selection
   - Develop value estimation for different trading strategies
   - Build exploration vs. exploitation mechanisms for strategy selection

2. **Feedback Processing**
   - Implement explicit user feedback integration
   - Create outcome tracking for trade signals
   - Develop automated performance evaluation
   - Build feedback aggregation and prioritization

3. **Performance Tracking and Metrics**
   - Implement comprehensive performance metrics
   - Create visualization and reporting capabilities
   - Develop trend analysis for performance over time
   - Build comparative analysis against benchmarks

4. **Pattern Recognition**
   - Implement pattern detection in successful trading strategies
   - Create correlation analysis between signals and outcomes
   - Develop anomaly detection for unusual market behavior
   - Build predictive modeling for strategy effectiveness

5. **Learning Integration**
   - Create a unified learning manager that coordinates all learning systems
   - Implement integration with memory systems
   - Develop mechanisms for applying learned patterns to new situations
   - Build continuous learning processes that run in the background

## Implementation Guidelines

- Use a modular approach with clear interfaces between components
- Implement proper abstraction layers for reinforcement learning algorithms
- Ensure all learning operations are asynchronous for performance
- Implement proper error handling and fallback mechanisms
- Create comprehensive unit tests for all learning components
- Document all learning interfaces and implementation details
- Consider performance implications of learning operations

## Technical Approach

### Reinforcement Learning Framework

```python
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import random

class ReinforcementLearning:
    """Reinforcement learning system for the TAAT AI Agent."""
    
    def __init__(self, config):
        """Initialize the reinforcement learning system."""
        self.config = config
        self.learning_rate = config.get("learning_rate", 0.1)
        self.discount_factor = config.get("discount_factor", 0.9)
        self.exploration_rate = config.get("exploration_rate", 0.2)
        self.min_exploration_rate = config.get("min_exploration_rate", 0.01)
        self.exploration_decay = config.get("exploration_decay", 0.995)
        self.q_values = {}  # State-action values
        
    async def update_q_value(self, state: str, action: str, reward: float, next_state: str) -> float:
        """
        Update Q-value for a state-action pair using Q-learning.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            
        Returns:
            Updated Q-value
        """
        # Get current Q-value
        state_key = self._get_state_key(state)
        action_key = self._get_action_key(action)
        current_q = self.q_values.get((state_key, action_key), 0.0)
        
        # Get max Q-value for next state
        next_state_key = self._get_state_key(next_state)
        next_q_values = [
            self.q_values.get((next_state_key, a), 0.0)
            for a in self._get_possible_actions(next_state)
        ]
        max_next_q = max(next_q_values) if next_q_values else 0.0
        
        # Update Q-value
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        
        # Store updated Q-value
        self.q_values[(state_key, action_key)] = new_q
        
        # Decay exploration rate
        self.exploration_rate = max(
            self.min_exploration_rate,
            self.exploration_rate * self.exploration_decay
        )
        
        return new_q
    
    async def select_action(self, state: str, possible_actions: List[str]) -> str:
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            state: Current state
            possible_actions: List of possible actions
            
        Returns:
            Selected action
        """
        # Exploration: random action
        if random.random() < self.exploration_rate:
            return random.choice(possible_actions)
        
        # Exploitation: best action
        state_key = self._get_state_key(state)
        q_values = [
            (action, self.q_values.get((state_key, self._get_action_key(action)), 0.0))
            for action in possible_actions
        ]
        
        # Sort by Q-value (descending)
        q_values.sort(key=lambda x: x[1], reverse=True)
        
        # Return action with highest Q-value
        return q_values[0][0]
    
    def _get_state_key(self, state: str) -> str:
        """Get a key for the state."""
        return f"state:{state}"
    
    def _get_action_key(self, action: str) -> str:
        """Get a key for the action."""
        return f"action:{action}"
    
    def _get_possible_actions(self, state: str) -> List[str]:
        """Get possible actions for a state."""
        # In a real implementation, this would be more sophisticated
        state_key = self._get_state_key(state)
        actions = set()
        
        for (s, a) in self.q_values.keys():
            if s == state_key:
                actions.add(a.replace("action:", ""))
        
        return list(actions) if actions else []
```

### Feedback Processing

```python
class FeedbackProcessor:
    """Feedback processing system for the TAAT AI Agent."""
    
    def __init__(self, db_manager):
        """Initialize the feedback processor."""
        self.db_manager = db_manager
        
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
        feedback_value = feedback_data.get("value", 0)
        feedback_text = feedback_data.get("text", "")
        
        # Store feedback
        feedback_id = await self.db_manager.store_feedback(
            feedback_type, feedback_value, feedback_text
        )
        
        # Update relevant models based on feedback type
        if feedback_type == "trade_signal":
            await self._update_signal_model(feedback_data)
        elif feedback_type == "trader_reliability":
            await self._update_trader_model(feedback_data)
        elif feedback_type == "strategy":
            await self._update_strategy_model(feedback_data)
        
        return {
            "feedback_id": feedback_id,
            "status": "processed",
            "applied_to": feedback_type
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
        trade_id = trade_data.get("trade_id")
        symbol = trade_data.get("symbol")
        action = trade_data.get("action")
        outcome = trade_data.get("outcome")
        profit_loss = trade_data.get("profit_loss", 0)
        
        # Calculate reward based on outcome
        reward = self._calculate_reward(outcome, profit_loss)
        
        # Store outcome
        outcome_id = await self.db_manager.store_trade_outcome(
            trade_id, symbol, action, outcome, profit_loss, reward
        )
        
        # Update models based on outcome
        await self._update_signal_model({
            "symbol": symbol,
            "action": action,
            "outcome": outcome,
            "reward": reward
        })
        
        if "trader_id" in trade_data:
            await self._update_trader_model({
                "trader_id": trade_data["trader_id"],
                "outcome": outcome,
                "reward": reward
            })
        
        return {
            "outcome_id": outcome_id,
            "reward": reward,
            "status": "processed"
        }
    
    def _calculate_reward(self, outcome: str, profit_loss: float) -> float:
        """Calculate reward based on outcome and profit/loss."""
        if outcome == "success":
            # Positive reward proportional to profit
            return min(1.0, max(0.1, profit_loss / 100))
        elif outcome == "failure":
            # Negative reward proportional to loss
            return max(-1.0, min(-0.1, profit_loss / 100))
        else:
            # Neutral reward for unknown outcome
            return 0.0
    
    async def _update_signal_model(self, data: Dict[str, Any]) -> None:
        """Update signal model based on feedback."""
        # Implementation depends on specific models
        pass
    
    async def _update_trader_model(self, data: Dict[str, Any]) -> None:
        """Update trader model based on feedback."""
        # Implementation depends on specific models
        pass
    
    async def _update_strategy_model(self, data: Dict[str, Any]) -> None:
        """Update strategy model based on feedback."""
        # Implementation depends on specific models
        pass
```

### Performance Metrics

```python
class PerformanceTracker:
    """Performance tracking system for the TAAT AI Agent."""
    
    def __init__(self, db_manager):
        """Initialize the performance tracker."""
        self.db_manager = db_manager
        
    async def calculate_metrics(self, timeframe: str = "all") -> Dict[str, Any]:
        """
        Calculate performance metrics.
        
        Args:
            timeframe: Time frame for metrics (all, day, week, month)
            
        Returns:
            Performance metrics
        """
        # Get trade outcomes for the specified timeframe
        outcomes = await self.db_manager.get_trade_outcomes(timeframe)
        
        # Calculate basic metrics
        total_trades = len(outcomes)
        successful_trades = sum(1 for o in outcomes if o["outcome"] == "success")
        failed_trades = sum(1 for o in outcomes if o["outcome"] == "failure")
        
        success_rate = successful_trades / total_trades if total_trades > 0 else 0
        
        # Calculate profit/loss metrics
        total_profit = sum(o["profit_loss"] for o in outcomes if o["profit_loss"] > 0)
        total_loss = sum(o["profit_loss"] for o in outcomes if o["profit_loss"] < 0)
        net_profit = total_profit + total_loss
        
        # Calculate advanced metrics
        avg_profit_per_trade = total_profit / successful_trades if successful_trades > 0 else 0
        avg_loss_per_trade = total_loss / failed_trades if failed_trades > 0 else 0
        
        profit_factor = abs(total_profit / total_loss) if total_loss != 0 else float('inf')
        
        # Return metrics
        return {
            "timeframe": timeframe,
            "total_trades": total_trades,
            "successful_trades": successful_trades,
            "failed_trades": failed_trades,
            "success_rate": success_rate,
            "total_profit": total_profit,
            "total_loss": total_loss,
            "net_profit": net_profit,
            "avg_profit_per_trade": avg_profit_per_trade,
            "avg_loss_per_trade": avg_loss_per_trade,
            "profit_factor": profit_factor
        }
    
    async def calculate_trader_metrics(self, trader_id: str, timeframe: str = "all") -> Dict[str, Any]:
        """
        Calculate performance metrics for a specific trader.
        
        Args:
            trader_id: Trader ID
            timeframe: Time frame for metrics
            
        Returns:
            Trader performance metrics
        """
        # Get trade outcomes for the specified trader and timeframe
        outcomes = await self.db_manager.get_trader_outcomes(trader_id, timeframe)
        
        # Calculate metrics (similar to calculate_metrics)
        # ...
        
        return {
            "trader_id": trader_id,
            "timeframe": timeframe,
            # Other metrics
        }
    
    async def generate_performance_report(self, timeframe: str = "all") -> Dict[str, Any]:
        """
        Generate a comprehensive performance report.
        
        Args:
            timeframe: Time frame for the report
            
        Returns:
            Performance report
        """
        # Calculate overall metrics
        overall_metrics = await self.calculate_metrics(timeframe)
        
        # Calculate metrics by trader
        trader_ids = await self.db_manager.get_active_trader_ids(timeframe)
        trader_metrics = {}
        
        for trader_id in trader_ids:
            trader_metrics[trader_id] = await self.calculate_trader_metrics(trader_id, timeframe)
        
        # Calculate metrics by symbol
        symbols = await self.db_manager.get_active_symbols(timeframe)
        symbol_metrics = {}
        
        for symbol in symbols:
            symbol_metrics[symbol] = await self.calculate_symbol_metrics(symbol, timeframe)
        
        # Return comprehensive report
        return {
            "overall": overall_metrics,
            "by_trader": trader_metrics,
            "by_symbol": symbol_metrics,
            "generated_at": datetime.now().isoformat()
        }
```

### Pattern Recognition

```python
class PatternRecognition:
    """Pattern recognition system for the TAAT AI Agent."""
    
    def __init__(self, db_manager, episodic_memory):
        """Initialize the pattern recognition system."""
        self.db_manager = db_manager
        self.episodic_memory = episodic_memory
        
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
            data = await self.db_manager.get_trade_outcomes(timeframe)
        elif data_type == "signals":
            data = await self.db_manager.get_trade_signals(timeframe)
        else:
            raise ValueError(f"Unsupported data type: {data_type}")
        
        # Detect patterns based on data type
        if data_type == "trades":
            return await self._detect_trade_patterns(data)
        elif data_type == "signals":
            return await self._detect_signal_patterns(data)
        
        return []
    
    async def _detect_trade_patterns(self, trades: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect patterns in trade data."""
        patterns = []
        
        # Group trades by symbol
        trades_by_symbol = {}
        for trade in trades:
            symbol = trade.get("symbol")
            if symbol not in trades_by_symbol:
                trades_by_symbol[symbol] = []
            trades_by_symbol[symbol].append(trade)
        
        # Analyze each symbol
        for symbol, symbol_trades in trades_by_symbol.items():
            # Calculate success rate
            total = len(symbol_trades)
            successful = sum(1 for t in symbol_trades if t.get("outcome") == "success")
            success_rate = successful / total if total > 0 else 0
            
            # Check if this is a high-success pattern
            if success_rate >= 0.7 and total >= 5:
                patterns.append({
                    "type": "high_success_symbol",
                    "symbol": symbol,
                    "success_rate": success_rate,
                    "sample_size": total,
                    "confidence": min(1.0, total / 10) * success_rate
                })
        
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
            successful = sum(1 for t in trader_trades if t.get("outcome") == "success")
            success_rate = successful / total if total > 0 else 0
            
            # Check if this is a reliable trader pattern
            if success_rate >= 0.7 and total >= 5:
                patterns.append({
                    "type": "reliable_trader",
                    "trader_id": trader_id,
                    "success_rate": success_rate,
                    "sample_size": total,
                    "confidence": min(1.0, total / 10) * success_rate
                })
        
        return patterns
    
    async def _detect_signal_patterns(self, signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect patterns in signal data."""
        # Similar to _detect_trade_patterns but for signals
        return []
```

### Learning Manager

```python
class LearningManager:
    """Learning manager for the TAAT AI Agent."""
    
    def __init__(self, config, db_manager, memory_manager):
        """Initialize the learning manager."""
        self.config = config
        self.db_manager = db_manager
        self.memory_manager = memory_manager
        
        # Initialize learning systems
        self.reinforcement_learning = ReinforcementLearning(config.get("reinforcement_learning", {}))
        self.feedback_processor = FeedbackProcessor(db_manager)
        self.performance_tracker = PerformanceTracker(db_manager)
        self.pattern_recognition = PatternRecognition(db_manager, memory_manager.episodic_memory)
        
    async def process_feedback(self, feedback_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process feedback and update learning systems.
        
        Args:
            feedback_data: Feedback data
            
        Returns:
            Processing result
        """
        # Process feedback
        result = await self.feedback_processor.process_user_feedback(feedback_data)
        
        # Update reinforcement learning if applicable
        if "state" in feedback_data and "action" in feedback_data:
            await self.reinforcement_learning.update_q_value(
                feedback_data["state"],
                feedback_data["action"],
                feedback_data.get("reward", 0.0),
                feedback_data.get("next_state", "")
            )
        
        return result
    
    async def process_outcome(self, outcome_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process outcome and update learning systems.
        
        Args:
            outcome_data: Outcome data
            
        Returns:
            Processing result
        """
        # Process outcome
        result = await self.feedback_processor.process_trade_outcome(outcome_data)
        
        # Update reinforcement learning if applicable
        if "state" in outcome_data and "action" in outcome_data:
            await self.reinforcement_learning.update_q_value(
                outcome_data["state"],
                outcome_data["action"],
                result.get("reward", 0.0),
                outcome_data.get("next_state", "")
            )
        
        return result
    
    async def select_action(self, state: str, possible_actions: List[str]) -> str:
        """
        Select an action using learned policies.
        
        Args:
            state: Current state
            possible_actions: List of possible actions
            
        Returns:
            Selected action
        """
        return await self.reinforcement_learning.select_action(state, possible_actions)
    
    async def get_performance_metrics(self, timeframe: str = "all") -> Dict[str, Any]:
        """
        Get performance metrics.
        
        Args:
            timeframe: Time frame for metrics
            
        Returns:
            Performance metrics
        """
        return await self.performance_tracker.calculate_metrics(timeframe)
    
    async def get_detected_patterns(self, data_type: str = "trades") -> List[Dict[str, Any]]:
        """
        Get detected patterns.
        
        Args:
            data_type: Type of data to analyze
            
        Returns:
            Detected patterns
        """
        return await self.pattern_recognition.detect_patterns(data_type)
    
    async def run_learning_cycle(self) -> Dict[str, Any]:
        """
        Run a complete learning cycle.
        
        Returns:
            Learning cycle results
        """
        # Update performance metrics
        metrics = await self.performance_tracker.calculate_metrics()
        
        # Detect patterns
        trade_patterns = await self.pattern_recognition.detect_patterns("trades")
        signal_patterns = await self.pattern_recognition.detect_patterns("signals")
        
        # Store patterns in procedural memory
        for pattern in trade_patterns:
            await self.memory_manager.procedural_memory.store_pattern(
                pattern["type"],
                pattern,
                success=True
            )
        
        for pattern in signal_patterns:
            await self.memory_manager.procedural_memory.store_pattern(
                pattern["type"],
                pattern,
                success=True
            )
        
        return {
            "metrics": metrics,
            "patterns": {
                "trades": trade_patterns,
                "signals": signal_patterns
            }
        }
```

## Integration Considerations

- The learning systems should integrate with the existing agent architecture and memory systems
- The `EnhancedTaatAgent` class should be updated to use the new learning systems
- The memory manager should be enhanced to support learning operations
- Consider the performance implications of learning operations
- Implement background learning processes that run periodically

## Evaluation Criteria

- Learning effectiveness (agent improves over time)
- Feedback integration (user feedback improves agent behavior)
- Performance tracking accuracy (metrics reflect actual performance)
- Pattern recognition quality (detected patterns are meaningful)
- Code quality, documentation, and test coverage

## Deliverables

1. Reinforcement learning implementation
2. Feedback processing system
3. Performance tracking and metrics
4. Pattern recognition system
5. Learning manager for coordinated learning
6. Integration with existing agent architecture
7. Comprehensive tests for all learning components
8. Documentation of learning architecture and usage

## Future Considerations

This phase focuses on the learning systems themselves. In future phases, we will:
- Enhance the cognition module to better utilize learned patterns
- Implement more sophisticated reinforcement learning algorithms
- Develop visualization tools for learning progress
- Create explainable AI mechanisms for learning decisions
