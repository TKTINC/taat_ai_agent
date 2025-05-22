"""
Learning Manager for integrating all learning systems.

This module provides a unified learning manager that coordinates all learning systems,
including reinforcement learning, feedback processing, performance tracking, and pattern recognition.
"""

import json
import asyncio
from typing import Dict, List, Any, Optional, Tuple

from src.memory_systems.manager import MemoryManager
from src.learning_systems.config import LearningConfig
from src.learning_systems.reinforcement import ReinforcementLearning
from src.learning_systems.feedback import FeedbackProcessor
from src.learning_systems.performance import PerformanceTracker
from src.learning_systems.pattern import PatternRecognition


class LearningManager:
    """
    Learning Manager for integrating all learning systems.
    
    Coordinates reinforcement learning, feedback processing, performance tracking,
    and pattern recognition to provide a unified interface for learning operations.
    """
    
    def __init__(self, config: LearningConfig, db_manager, memory_manager: MemoryManager):
        """
        Initialize the learning manager.
        
        Args:
            config: Learning configuration
            db_manager: Database manager
            memory_manager: Memory manager
        """
        self.config = config
        self.db_manager = db_manager
        self.memory_manager = memory_manager
        self.learning_cycle_interval = config.learning_cycle_interval
        self.background_learning = config.background_learning
        
        # Initialize learning systems
        self.reinforcement_learning = ReinforcementLearning(config.reinforcement_learning)
        self.feedback_processor = FeedbackProcessor(config.feedback, db_manager)
        self.performance_tracker = PerformanceTracker(config.performance, db_manager)
        self.pattern_recognition = PatternRecognition(
            config.pattern, db_manager, memory_manager.episodic_memory
        )
        
        # Background learning task
        self.background_task = None
        
        # Start background learning if enabled
        if self.background_learning:
            self.start_background_learning()
    
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
                feedback_data.get("value", 0.0),
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
    
    async def select_action(self, state: Any, possible_actions: List[Any]) -> Any:
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
    
    async def get_relevant_patterns(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Get patterns relevant to the current context.
        
        Args:
            context: Current context
            
        Returns:
            List of relevant patterns
        """
        return await self.pattern_recognition.get_relevant_patterns(context)
    
    async def run_learning_cycle(self) -> Dict[str, Any]:
        """
        Run a complete learning cycle.
        
        Returns:
            Learning cycle results
        """
        results = {}
        
        try:
            # Update performance metrics
            metrics = await self.performance_tracker.calculate_metrics()
            results["metrics"] = metrics
            
            # Check if performance report should be generated
            if await self.performance_tracker.should_generate_report():
                report = await self.performance_tracker.generate_performance_report()
                results["performance_report"] = report
            
            # Detect patterns
            trade_patterns = await self.pattern_recognition.detect_patterns("trades")
            signal_patterns = await self.pattern_recognition.detect_patterns("signals")
            feedback_patterns = await self.pattern_recognition.detect_patterns("feedback")
            
            results["patterns"] = {
                "trades": trade_patterns,
                "signals": signal_patterns,
                "feedback": feedback_patterns
            }
            
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
            
            for pattern in feedback_patterns:
                await self.memory_manager.procedural_memory.store_pattern(
                    pattern["type"],
                    pattern,
                    success=True
                )
            
        except Exception as e:
            # Log error
            print(f"Error running learning cycle: {e}")
            results["error"] = str(e)
        
        return results
    
    def start_background_learning(self) -> None:
        """Start background learning task."""
        if self.background_task is None:
            self.background_task = asyncio.create_task(self._background_learning_loop())
    
    def stop_background_learning(self) -> None:
        """Stop background learning task."""
        if self.background_task is not None:
            self.background_task.cancel()
            self.background_task = None
    
    async def _background_learning_loop(self) -> None:
        """Background learning loop."""
        while True:
            try:
                # Run learning cycle
                await self.run_learning_cycle()
                
                # Wait for next cycle
                await asyncio.sleep(self.learning_cycle_interval)
            except asyncio.CancelledError:
                # Task was cancelled
                break
            except Exception as e:
                # Log error and continue
                print(f"Error in background learning loop: {e}")
                await asyncio.sleep(60)  # Wait a bit before retrying
