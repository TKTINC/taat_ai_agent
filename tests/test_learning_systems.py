"""
Tests for the learning systems module.

This module provides tests for the learning systems, including reinforcement learning,
feedback processing, performance tracking, pattern recognition, and the learning manager.
"""

import os
import json
import pytest
import datetime
from unittest.mock import MagicMock, patch, AsyncMock

from src.learning_systems.config import (
    LearningConfig, ReinforcementLearningConfig, FeedbackConfig,
    PerformanceConfig, PatternConfig
)
from src.learning_systems.reinforcement import ReinforcementLearning
from src.learning_systems.feedback import FeedbackProcessor
from src.learning_systems.performance import PerformanceTracker
from src.learning_systems.pattern import PatternRecognition
from src.learning_systems.manager import LearningManager
from src.learning_systems.integration import LearningTaatAgent


# Test fixtures
@pytest.fixture
def reinforcement_config():
    """Fixture for reinforcement learning configuration."""
    return ReinforcementLearningConfig(
        learning_rate=0.1,
        discount_factor=0.9,
        exploration_rate=0.2,
        min_exploration_rate=0.01,
        exploration_decay=0.995,
        reward_scale=1.0
    )


@pytest.fixture
def feedback_config():
    """Fixture for feedback configuration."""
    return FeedbackConfig(
        positive_threshold=0.7,
        negative_threshold=0.3,
        feedback_weight=0.8,
        outcome_weight=0.6,
        feedback_decay=0.9
    )


@pytest.fixture
def performance_config():
    """Fixture for performance configuration."""
    return PerformanceConfig(
        metrics_window_size=100,
        min_sample_size=5,
        confidence_threshold=0.6,
        report_frequency=24
    )


@pytest.fixture
def pattern_config():
    """Fixture for pattern configuration."""
    return PatternConfig(
        min_pattern_occurrences=5,
        min_pattern_confidence=0.7,
        max_patterns_per_type=10,
        pattern_similarity_threshold=0.8
    )


@pytest.fixture
def learning_config(reinforcement_config, feedback_config, performance_config, pattern_config):
    """Fixture for learning configuration."""
    return LearningConfig(
        reinforcement_learning=reinforcement_config,
        feedback=feedback_config,
        performance=performance_config,
        pattern=pattern_config,
        learning_cycle_interval=3600,
        background_learning=True
    )


@pytest.fixture
def mock_db_manager():
    """Fixture for mock database manager."""
    db_manager = AsyncMock()
    
    # Mock methods
    db_manager.store_feedback = AsyncMock(return_value="feedback_id_123")
    db_manager.store_trade_outcome = AsyncMock(return_value="outcome_id_123")
    db_manager.get_market_knowledge = AsyncMock(return_value=None)
    db_manager.create_market_knowledge = AsyncMock(return_value={"data": {}})
    db_manager.update_market_knowledge = AsyncMock()
    db_manager.get_trader_profile = AsyncMock(return_value=None)
    db_manager.create_trader_profile = AsyncMock(return_value={"data": {}})
    db_manager.update_trader_profile = AsyncMock()
    db_manager.get_trade_outcomes = AsyncMock(return_value=[])
    db_manager.get_trader_outcomes = AsyncMock(return_value=[])
    db_manager.get_symbol_outcomes = AsyncMock(return_value=[])
    db_manager.get_active_trader_ids = AsyncMock(return_value=[])
    db_manager.get_active_symbols = AsyncMock(return_value=[])
    db_manager.get_historical_metrics = AsyncMock(return_value=[])
    db_manager.store_metrics = AsyncMock()
    db_manager.store_trader_metrics = AsyncMock()
    db_manager.store_symbol_metrics = AsyncMock()
    db_manager.store_performance_report = AsyncMock()
    db_manager.get_trade_signals = AsyncMock(return_value=[])
    db_manager.get_feedback_history = AsyncMock(return_value=[])
    db_manager.get_patterns_by_type = AsyncMock(return_value=[])
    db_manager.store_pattern = AsyncMock()
    
    return db_manager


@pytest.fixture
def mock_memory_manager():
    """Fixture for mock memory manager."""
    memory_manager = AsyncMock()
    
    # Mock episodic memory
    memory_manager.episodic_memory = AsyncMock()
    memory_manager.episodic_memory.retrieve_similar_experiences = AsyncMock(return_value=[])
    
    # Mock procedural memory
    memory_manager.procedural_memory = AsyncMock()
    memory_manager.procedural_memory.store_pattern = AsyncMock()
    
    return memory_manager


# Tests for ReinforcementLearning
class TestReinforcementLearning:
    """Tests for the ReinforcementLearning class."""
    
    @pytest.mark.asyncio
    async def test_initialization(self, reinforcement_config):
        """Test initialization of reinforcement learning."""
        rl = ReinforcementLearning(reinforcement_config)
        
        assert rl.learning_rate == reinforcement_config.learning_rate
        assert rl.discount_factor == reinforcement_config.discount_factor
        assert rl.exploration_rate == reinforcement_config.exploration_rate
        assert rl.min_exploration_rate == reinforcement_config.min_exploration_rate
        assert rl.exploration_decay == reinforcement_config.exploration_decay
        assert rl.reward_scale == reinforcement_config.reward_scale
        assert rl.q_values == {}
        assert rl.state_transitions == {}
        assert rl.action_history == []
    
    @pytest.mark.asyncio
    async def test_update_q_value(self, reinforcement_config):
        """Test updating Q-value."""
        rl = ReinforcementLearning(reinforcement_config)
        
        state = "state1"
        action = "action1"
        reward = 1.0
        next_state = "state2"
        
        # Update Q-value
        new_q = await rl.update_q_value(state, action, reward, next_state)
        
        # Check Q-value
        state_key = rl._get_state_key(state)
        action_key = rl._get_action_key(action)
        state_action_key = rl._get_state_action_key(state_key, action_key)
        
        assert state_action_key in rl.q_values
        assert rl.q_values[state_action_key] == new_q
        
        # Check state transition
        next_state_key = rl._get_state_key(next_state)
        transition_key = f"{state_key}|{next_state_key}"
        
        assert transition_key in rl.state_transitions
        assert rl.state_transitions[transition_key] == 1
    
    @pytest.mark.asyncio
    async def test_select_action_exploration(self, reinforcement_config):
        """Test selecting action with exploration."""
        rl = ReinforcementLearning(reinforcement_config)
        rl.exploration_rate = 1.0  # Force exploration
        
        state = "state1"
        possible_actions = ["action1", "action2", "action3"]
        
        # Select action
        selected_action = await rl.select_action(state, possible_actions)
        
        # Check selected action
        assert selected_action in possible_actions
        assert len(rl.action_history) == 1
        assert rl.action_history[0]["type"] == "exploration"
    
    @pytest.mark.asyncio
    async def test_select_action_exploitation(self, reinforcement_config):
        """Test selecting action with exploitation."""
        rl = ReinforcementLearning(reinforcement_config)
        rl.exploration_rate = 0.0  # Force exploitation
        
        state = "state1"
        possible_actions = ["action1", "action2", "action3"]
        
        # Set Q-values
        state_key = rl._get_state_key(state)
        action_key1 = rl._get_action_key("action1")
        action_key2 = rl._get_action_key("action2")
        action_key3 = rl._get_action_key("action3")
        
        rl.q_values[rl._get_state_action_key(state_key, action_key1)] = 0.5
        rl.q_values[rl._get_state_action_key(state_key, action_key2)] = 0.8
        rl.q_values[rl._get_state_action_key(state_key, action_key3)] = 0.2
        
        # Select action
        selected_action = await rl.select_action(state, possible_actions)
        
        # Check selected action (should be action2 with highest Q-value)
        assert selected_action == "action2"
        assert len(rl.action_history) == 1
        assert rl.action_history[0]["type"] == "exploitation"
    
    @pytest.mark.asyncio
    async def test_save_load_model(self, reinforcement_config):
        """Test saving and loading model."""
        rl = ReinforcementLearning(reinforcement_config)
        
        # Set some Q-values and state transitions
        rl.q_values = {"state1|action1": 0.5, "state1|action2": 0.8}
        rl.state_transitions = {"state1|state2": 2, "state2|state3": 1}
        rl.exploration_rate = 0.15
        
        # Save model
        model_data = rl.save_model()
        
        # Create new instance
        rl2 = ReinforcementLearning(reinforcement_config)
        
        # Load model
        rl2.load_model(model_data)
        
        # Check loaded data
        assert rl2.q_values == rl.q_values
        assert rl2.state_transitions == rl.state_transitions
        assert rl2.exploration_rate == rl.exploration_rate


# Tests for FeedbackProcessor
class TestFeedbackProcessor:
    """Tests for the FeedbackProcessor class."""
    
    @pytest.mark.asyncio
    async def test_initialization(self, feedback_config, mock_db_manager):
        """Test initialization of feedback processor."""
        fp = FeedbackProcessor(feedback_config, mock_db_manager)
        
        assert fp.positive_threshold == feedback_config.positive_threshold
        assert fp.negative_threshold == feedback_config.negative_threshold
        assert fp.feedback_weight == feedback_config.feedback_weight
        assert fp.outcome_weight == feedback_config.outcome_weight
        assert fp.feedback_decay == feedback_config.feedback_decay
        assert fp.feedback_history == []
    
    @pytest.mark.asyncio
    async def test_process_user_feedback(self, feedback_config, mock_db_manager):
        """Test processing user feedback."""
        fp = FeedbackProcessor(feedback_config, mock_db_manager)
        
        # Create feedback data
        feedback_data = {
            "type": "trade_signal",
            "value": 0.8,
            "text": "Good signal",
            "symbol": "AAPL"
        }
        
        # Process feedback
        result = await fp.process_user_feedback(feedback_data)
        
        # Check result
        assert result["status"] == "processed"
        assert result["applied_to"] == "trade_signal"
        assert result["normalized_value"] == 0.8
        assert "feedback_id" in result
        
        # Check feedback history
        assert len(fp.feedback_history) == 1
        assert fp.feedback_history[0]["type"] == "trade_signal"
        assert fp.feedback_history[0]["value"] == 0.8
        assert fp.feedback_history[0]["text"] == "Good signal"
        assert fp.feedback_history[0]["source"] == "user"
        
        # Check database calls
        mock_db_manager.store_feedback.assert_called_once()
        mock_db_manager.get_market_knowledge.assert_called_once_with("AAPL")
    
    @pytest.mark.asyncio
    async def test_process_trade_outcome(self, feedback_config, mock_db_manager):
        """Test processing trade outcome."""
        fp = FeedbackProcessor(feedback_config, mock_db_manager)
        
        # Create trade data
        trade_data = {
            "trade_id": "trade123",
            "symbol": "AAPL",
            "action": "buy",
            "outcome": "success",
            "profit_loss": 50.0,
            "trader_id": "trader123"
        }
        
        # Process outcome
        result = await fp.process_trade_outcome(trade_data)
        
        # Check result
        assert result["status"] == "processed"
        assert "reward" in result
        assert result["reward"] > 0  # Positive reward for success
        assert "outcome_id" in result
        
        # Check feedback history
        assert len(fp.feedback_history) == 1
        assert fp.feedback_history[0]["trade_id"] == "trade123"
        assert fp.feedback_history[0]["symbol"] == "AAPL"
        assert fp.feedback_history[0]["action"] == "buy"
        assert fp.feedback_history[0]["outcome"] == "success"
        assert fp.feedback_history[0]["profit_loss"] == 50.0
        assert fp.feedback_history[0]["source"] == "system"
        
        # Check database calls
        mock_db_manager.store_trade_outcome.assert_called_once()
        mock_db_manager.get_market_knowledge.assert_called_once_with("AAPL")
        mock_db_manager.get_trader_profile.assert_called_once_with("trader123")


# Tests for PerformanceTracker
class TestPerformanceTracker:
    """Tests for the PerformanceTracker class."""
    
    @pytest.mark.asyncio
    async def test_initialization(self, performance_config, mock_db_manager):
        """Test initialization of performance tracker."""
        pt = PerformanceTracker(performance_config, mock_db_manager)
        
        assert pt.metrics_window_size == performance_config.metrics_window_size
        assert pt.min_sample_size == performance_config.min_sample_size
        assert pt.confidence_threshold == performance_config.confidence_threshold
        assert pt.report_frequency == performance_config.report_frequency
        assert pt.metrics_history == []
        assert pt.last_report_time is None
    
    @pytest.mark.asyncio
    async def test_calculate_metrics_insufficient_data(self, performance_config, mock_db_manager):
        """Test calculating metrics with insufficient data."""
        pt = PerformanceTracker(performance_config, mock_db_manager)
        
        # Mock empty trade outcomes
        mock_db_manager.get_trade_outcomes.return_value = []
        
        # Calculate metrics
        metrics = await pt.calculate_metrics()
        
        # Check metrics
        assert metrics["status"] == "insufficient_data"
        assert metrics["total_trades"] == 0
        assert metrics["min_sample_size"] == performance_config.min_sample_size
    
    @pytest.mark.asyncio
    async def test_calculate_metrics_with_data(self, performance_config, mock_db_manager):
        """Test calculating metrics with sufficient data."""
        pt = PerformanceTracker(performance_config, mock_db_manager)
        
        # Mock trade outcomes
        mock_db_manager.get_trade_outcomes.return_value = [
            {"outcome": "success", "profit_loss": 50.0},
            {"outcome": "success", "profit_loss": 30.0},
            {"outcome": "failure", "profit_loss": -20.0},
            {"outcome": "success", "profit_loss": 40.0},
            {"outcome": "failure", "profit_loss": -10.0},
            {"outcome": "unknown", "profit_loss": 0.0}
        ]
        
        # Calculate metrics
        metrics = await pt.calculate_metrics()
        
        # Check metrics
        assert "status" not in metrics  # No status means success
        assert metrics["total_trades"] == 6
        assert metrics["successful_trades"] == 3
        assert metrics["failed_trades"] == 2
        assert metrics["neutral_trades"] == 1
        assert metrics["success_rate"] == 0.5
        assert metrics["total_profit"] == 120.0
        assert metrics["total_loss"] == -30.0
        assert metrics["net_profit"] == 90.0
        
        # Check metrics history
        assert len(pt.metrics_history) == 1
        assert pt.metrics_history[0]["total_trades"] == 6
        
        # Check database calls
        mock_db_manager.store_metrics.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_generate_performance_report(self, performance_config, mock_db_manager):
        """Test generating performance report."""
        pt = PerformanceTracker(performance_config, mock_db_manager)
        
        # Mock methods
        pt.calculate_metrics = AsyncMock(return_value={"total_trades": 10})
        pt.calculate_trader_metrics = AsyncMock(return_value={"trader_id": "trader123", "total_trades": 5})
        pt.calculate_symbol_metrics = AsyncMock(return_value={"symbol": "AAPL", "total_trades": 3})
        pt._calculate_trend_metrics = AsyncMock(return_value={"success_rate_trend": 0.5})
        
        # Mock active traders and symbols
        mock_db_manager.get_active_trader_ids.return_value = ["trader123"]
        mock_db_manager.get_active_symbols.return_value = ["AAPL"]
        
        # Generate report
        report = await pt.generate_performance_report()
        
        # Check report
        assert report["overall"]["total_trades"] == 10
        assert "trader123" in report["by_trader"]
        assert report["by_trader"]["trader123"]["total_trades"] == 5
        assert "AAPL" in report["by_symbol"]
        assert report["by_symbol"]["AAPL"]["total_trades"] == 3
        assert report["trends"]["success_rate_trend"] == 0.5
        
        # Check last report time
        assert pt.last_report_time is not None
        
        # Check database calls
        mock_db_manager.store_performance_report.assert_called_once()


# Tests for PatternRecognition
class TestPatternRecognition:
    """Tests for the PatternRecognition class."""
    
    @pytest.mark.asyncio
    async def test_initialization(self, pattern_config, mock_db_manager, mock_memory_manager):
        """Test initialization of pattern recognition."""
        pr = PatternRecognition(pattern_config, mock_db_manager, mock_memory_manager.episodic_memory)
        
        assert pr.min_pattern_occurrences == pattern_config.min_pattern_occurrences
        assert pr.min_pattern_confidence == pattern_config.min_pattern_confidence
        assert pr.max_patterns_per_type == pattern_config.max_patterns_per_type
        assert pr.pattern_similarity_threshold == pattern_config.pattern_similarity_threshold
        assert pr.pattern_cache == {}
    
    @pytest.mark.asyncio
    async def test_detect_trade_patterns(self, pattern_config, mock_db_manager, mock_memory_manager):
        """Test detecting trade patterns."""
        pr = PatternRecognition(pattern_config, mock_db_manager, mock_memory_manager.episodic_memory)
        
        # Mock trade outcomes with pattern
        mock_db_manager.get_trade_outcomes.return_value = [
            {"symbol": "AAPL", "outcome": "success", "trader_id": "trader123"},
            {"symbol": "AAPL", "outcome": "success", "trader_id": "trader123"},
            {"symbol": "AAPL", "outcome": "success", "trader_id": "trader123"},
            {"symbol": "AAPL", "outcome": "success", "trader_id": "trader123"},
            {"symbol": "AAPL", "outcome": "failure", "trader_id": "trader123"},
            {"symbol": "GOOG", "outcome": "success", "trader_id": "trader456"},
            {"symbol": "GOOG", "outcome": "failure", "trader_id": "trader456"}
        ]
        
        # Detect patterns
        patterns = await pr.detect_patterns("trades")
        
        # Check patterns
        assert len(patterns) > 0
        
        # Find high success symbol pattern
        high_success_patterns = [p for p in patterns if p["type"] == "high_success_symbol"]
        assert len(high_success_patterns) > 0
        assert high_success_patterns[0]["symbol"] == "AAPL"
        assert high_success_patterns[0]["success_rate"] == 0.8  # 4 out of 5
        
        # Check database calls
        mock_db_manager.store_pattern.assert_called()
    
    @pytest.mark.asyncio
    async def test_get_relevant_patterns(self, pattern_config, mock_db_manager, mock_memory_manager):
        """Test getting relevant patterns."""
        pr = PatternRecognition(pattern_config, mock_db_manager, mock_memory_manager.episodic_memory)
        
        # Mock patterns
        mock_db_manager.get_patterns_by_type.side_effect = lambda pattern_type, **kwargs: {
            "reliable_trader": [
                {"type": "reliable_trader", "trader_id": "trader123", "confidence": 0.8},
                {"type": "reliable_trader", "trader_id": "trader456", "confidence": 0.7}
            ],
            "high_success_symbol": [
                {"type": "high_success_symbol", "symbol": "AAPL", "confidence": 0.9},
                {"type": "high_success_symbol", "symbol": "GOOG", "confidence": 0.6}
            ],
            "action_symbol_success": [
                {"type": "action_symbol_success", "action": "buy", "symbol": "AAPL", "confidence": 0.8},
                {"type": "action_symbol_success", "action": "sell", "symbol": "AAPL", "confidence": 0.7}
            ]
        }[pattern_type]
        
        # Get relevant patterns
        context = {
            "trader_id": "trader123",
            "symbol": "AAPL",
            "action": "buy"
        }
        
        patterns = await pr.get_relevant_patterns(context)
        
        # Check patterns
        assert len(patterns) > 0
        
        # Check trader pattern
        trader_patterns = [p for p in patterns if p["type"] == "reliable_trader"]
        assert len(trader_patterns) > 0
        assert trader_patterns[0]["trader_id"] == "trader123"
        
        # Check symbol pattern
        symbol_patterns = [p for p in patterns if p["type"] == "high_success_symbol"]
        assert len(symbol_patterns) > 0
        assert symbol_patterns[0]["symbol"] == "AAPL"
        
        # Check action-symbol pattern
        action_symbol_patterns = [p for p in patterns if p["type"] == "action_symbol_success"]
        assert len(action_symbol_patterns) > 0
        assert action_symbol_patterns[0]["action"] == "buy"
        assert action_symbol_patterns[0]["symbol"] == "AAPL"


# Tests for LearningManager
class TestLearningManager:
    """Tests for the LearningManager class."""
    
    @pytest.mark.asyncio
    async def test_initialization(self, learning_config, mock_db_manager, mock_memory_manager):
        """Test initialization of learning manager."""
        with patch("src.learning_systems.manager.ReinforcementLearning") as mock_rl, \
             patch("src.learning_systems.manager.FeedbackProcessor") as mock_fp, \
             patch("src.learning_systems.manager.PerformanceTracker") as mock_pt, \
             patch("src.learning_systems.manager.PatternRecognition") as mock_pr, \
             patch("src.learning_systems.manager.asyncio.create_task") as mock_create_task:
            
            # Create learning manager
            lm = LearningManager(learning_config, mock_db_manager, mock_memory_manager)
            
            # Check initialization
            assert lm.learning_cycle_interval == learning_config.learning_cycle_interval
            assert lm.background_learning == learning_config.background_learning
            
            # Check component initialization
            mock_rl.assert_called_once_with(learning_config.reinforcement_learning)
            mock_fp.assert_called_once_with(learning_config.feedback, mock_db_manager)
            mock_pt.assert_called_once_with(learning_config.performance, mock_db_manager)
            mock_pr.assert_called_once_with(
                learning_config.pattern, mock_db_manager, mock_memory_manager.episodic_memory
            )
            
            # Check background task
            if learning_config.background_learning:
                mock_create_task.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_process_feedback(self, learning_config, mock_db_manager, mock_memory_manager):
        """Test processing feedback."""
        with patch("src.learning_systems.manager.ReinforcementLearning") as mock_rl_class, \
             patch("src.learning_systems.manager.FeedbackProcessor") as mock_fp_class, \
             patch("src.learning_systems.manager.PerformanceTracker") as mock_pt_class, \
             patch("src.learning_systems.manager.PatternRecognition") as mock_pr_class, \
             patch("src.learning_systems.manager.asyncio.create_task"):
            
            # Mock components
            mock_rl = AsyncMock()
            mock_fp = AsyncMock()
            mock_pt = AsyncMock()
            mock_pr = AsyncMock()
            
            mock_rl_class.return_value = mock_rl
            mock_fp_class.return_value = mock_fp
            mock_pt_class.return_value = mock_pt
            mock_pr_class.return_value = mock_pr
            
            # Mock feedback processing
            mock_fp.process_user_feedback.return_value = {"status": "processed"}
            
            # Create learning manager
            lm = LearningManager(learning_config, mock_db_manager, mock_memory_manager)
            
            # Process feedback
            feedback_data = {
                "type": "trade_signal",
                "value": 0.8,
                "state": "state1",
                "action": "action1",
                "next_state": "state2"
            }
            
            result = await lm.process_feedback(feedback_data)
            
            # Check result
            assert result["status"] == "processed"
            
            # Check component calls
            mock_fp.process_user_feedback.assert_called_once_with(feedback_data)
            mock_rl.update_q_value.assert_called_once_with(
                feedback_data["state"],
                feedback_data["action"],
                feedback_data.get("value", 0.0),
                feedback_data.get("next_state", "")
            )
    
    @pytest.mark.asyncio
    async def test_run_learning_cycle(self, learning_config, mock_db_manager, mock_memory_manager):
        """Test running learning cycle."""
        with patch("src.learning_systems.manager.ReinforcementLearning") as mock_rl_class, \
             patch("src.learning_systems.manager.FeedbackProcessor") as mock_fp_class, \
             patch("src.learning_systems.manager.PerformanceTracker") as mock_pt_class, \
             patch("src.learning_systems.manager.PatternRecognition") as mock_pr_class, \
             patch("src.learning_systems.manager.asyncio.create_task"):
            
            # Mock components
            mock_rl = AsyncMock()
            mock_fp = AsyncMock()
            mock_pt = AsyncMock()
            mock_pr = AsyncMock()
            
            mock_rl_class.return_value = mock_rl
            mock_fp_class.return_value = mock_fp
            mock_pt_class.return_value = mock_pt
            mock_pr_class.return_value = mock_pr
            
            # Mock component methods
            mock_pt.calculate_metrics.return_value = {"total_trades": 10}
            mock_pt.should_generate_report.return_value = True
            mock_pt.generate_performance_report.return_value = {"overall": {"total_trades": 10}}
            
            mock_pr.detect_patterns.side_effect = lambda data_type: {
                "trades": [{"type": "high_success_symbol"}],
                "signals": [{"type": "frequent_signaler"}],
                "feedback": [{"type": "consistent_feedback"}]
            }[data_type]
            
            # Create learning manager
            lm = LearningManager(learning_config, mock_db_manager, mock_memory_manager)
            
            # Run learning cycle
            results = await lm.run_learning_cycle()
            
            # Check results
            assert "metrics" in results
            assert results["metrics"]["total_trades"] == 10
            assert "performance_report" in results
            assert results["performance_report"]["overall"]["total_trades"] == 10
            assert "patterns" in results
            assert len(results["patterns"]["trades"]) == 1
            assert len(results["patterns"]["signals"]) == 1
            assert len(results["patterns"]["feedback"]) == 1
            
            # Check component calls
            mock_pt.calculate_metrics.assert_called_once()
            mock_pt.should_generate_report.assert_called_once()
            mock_pt.generate_performance_report.assert_called_once()
            mock_pr.detect_patterns.assert_called()
            mock_memory_manager.procedural_memory.store_pattern.assert_called()


# Tests for LearningTaatAgent
class TestLearningTaatAgent:
    """Tests for the LearningTaatAgent class."""
    
    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test initialization of learning agent."""
        with patch("src.learning_systems.integration.EnhancedTaatAgent.__init__") as mock_init, \
             patch("src.learning_systems.integration.LearningManager") as mock_lm_class, \
             patch("src.learning_systems.integration.load_learning_config") as mock_load_config:
            
            # Mock config
            mock_config = MagicMock()
            mock_config.learning = MagicMock()
            mock_load_config.return_value = mock_config
            
            # Mock learning manager
            mock_lm = MagicMock()
            mock_lm_class.return_value = mock_lm
            
            # Create agent
            agent = LearningTaatAgent()
            
            # Check initialization
            mock_init.assert_called_once_with(mock_config)
            mock_lm_class.assert_called_once()
            assert agent.learning_manager == mock_lm
    
    @pytest.mark.asyncio
    async def test_process_input(self):
        """Test processing input with learning."""
        with patch("src.learning_systems.integration.EnhancedTaatAgent.__init__", return_value=None), \
             patch("src.learning_systems.integration.LearningManager") as mock_lm_class, \
             patch("src.learning_systems.integration.load_learning_config"):
            
            # Mock learning manager
            mock_lm = AsyncMock()
            mock_lm_class.return_value = mock_lm
            
            # Mock get_relevant_patterns
            mock_lm.get_relevant_patterns.return_value = [{"type": "high_success_symbol"}]
            
            # Create agent
            agent = LearningTaatAgent()
            
            # Mock agent components
            agent.perception = AsyncMock()
            agent.memory_manager = AsyncMock()
            agent.cognition = AsyncMock()
            agent.action = AsyncMock()
            agent.db_manager = AsyncMock()
            
            # Mock component methods
            agent.perception.process_input.return_value = {"text": "buy AAPL"}
            agent.memory_manager.get_context.return_value = {"history": []}
            agent.cognition.process.return_value = {"action": "buy", "symbol": "AAPL"}
            agent.action.execute.return_value = {
                "outcome": "success",
                "profit_loss": 50.0
            }
            
            # Process input
            input_data = "buy AAPL"
            result = await agent.process_input(input_data)
            
            # Check result
            assert result["outcome"] == "success"
            assert result["profit_loss"] == 50.0
            
            # Check component calls
            agent.perception.process_input.assert_called_once_with(input_data, "text")
            agent.memory_manager.get_context.assert_called_once()
            agent.cognition.process.assert_called_once()
            agent.action.execute.assert_called_once()
            agent.memory_manager.update_memories.assert_called_once()
            mock_lm.process_outcome.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_process_feedback(self):
        """Test processing feedback."""
        with patch("src.learning_systems.integration.EnhancedTaatAgent.__init__", return_value=None), \
             patch("src.learning_systems.integration.LearningManager") as mock_lm_class, \
             patch("src.learning_systems.integration.load_learning_config"):
            
            # Mock learning manager
            mock_lm = AsyncMock()
            mock_lm_class.return_value = mock_lm
            
            # Mock process_feedback
            mock_lm.process_feedback.return_value = {"status": "processed"}
            
            # Create agent
            agent = LearningTaatAgent()
            agent.learning_manager = mock_lm
            
            # Process feedback
            feedback_data = {"type": "trade_signal", "value": 0.8}
            result = await agent.process_feedback(feedback_data)
            
            # Check result
            assert result["status"] == "processed"
            
            # Check component calls
            mock_lm.process_feedback.assert_called_once_with(feedback_data)
