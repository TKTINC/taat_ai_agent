"""
Configuration for learning systems.

This module extends the memory configuration with settings for learning systems.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Literal

from src.memory_systems.config import EnhancedAgentConfig, MemoryConfig


@dataclass
class ReinforcementLearningConfig:
    """Configuration for reinforcement learning."""
    learning_rate: float = 0.1
    discount_factor: float = 0.9
    exploration_rate: float = 0.2
    min_exploration_rate: float = 0.01
    exploration_decay: float = 0.995
    reward_scale: float = 1.0


@dataclass
class FeedbackConfig:
    """Configuration for feedback processing."""
    positive_threshold: float = 0.7
    negative_threshold: float = 0.3
    feedback_weight: float = 0.8
    outcome_weight: float = 0.6
    feedback_decay: float = 0.9


@dataclass
class PerformanceConfig:
    """Configuration for performance tracking."""
    metrics_window_size: int = 100
    min_sample_size: int = 5
    confidence_threshold: float = 0.6
    report_frequency: int = 24  # hours


@dataclass
class PatternConfig:
    """Configuration for pattern recognition."""
    min_pattern_occurrences: int = 5
    min_pattern_confidence: float = 0.7
    max_patterns_per_type: int = 10
    pattern_similarity_threshold: float = 0.8


@dataclass
class LearningConfig:
    """Configuration for learning systems."""
    reinforcement_learning: ReinforcementLearningConfig = field(default_factory=ReinforcementLearningConfig)
    feedback: FeedbackConfig = field(default_factory=FeedbackConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    pattern: PatternConfig = field(default_factory=PatternConfig)
    learning_cycle_interval: int = 3600  # seconds
    background_learning: bool = True


@dataclass
class LearningAgentConfig(EnhancedAgentConfig):
    """Enhanced configuration for the TAAT Agent with learning systems."""
    learning: LearningConfig = field(default_factory=LearningConfig)


def load_learning_config() -> LearningAgentConfig:
    """
    Load learning configuration from environment variables.
    
    Returns:
        LearningAgentConfig: The loaded configuration
    """
    # Load enhanced config (with memory systems)
    enhanced_config = super(LearningAgentConfig, LearningAgentConfig).load_enhanced_config()
    
    # Reinforcement learning config
    rl_config = ReinforcementLearningConfig(
        learning_rate=float(os.environ.get("RL_LEARNING_RATE", "0.1")),
        discount_factor=float(os.environ.get("RL_DISCOUNT_FACTOR", "0.9")),
        exploration_rate=float(os.environ.get("RL_EXPLORATION_RATE", "0.2")),
        min_exploration_rate=float(os.environ.get("RL_MIN_EXPLORATION_RATE", "0.01")),
        exploration_decay=float(os.environ.get("RL_EXPLORATION_DECAY", "0.995")),
        reward_scale=float(os.environ.get("RL_REWARD_SCALE", "1.0"))
    )
    
    # Feedback config
    feedback_config = FeedbackConfig(
        positive_threshold=float(os.environ.get("FEEDBACK_POSITIVE_THRESHOLD", "0.7")),
        negative_threshold=float(os.environ.get("FEEDBACK_NEGATIVE_THRESHOLD", "0.3")),
        feedback_weight=float(os.environ.get("FEEDBACK_WEIGHT", "0.8")),
        outcome_weight=float(os.environ.get("OUTCOME_WEIGHT", "0.6")),
        feedback_decay=float(os.environ.get("FEEDBACK_DECAY", "0.9"))
    )
    
    # Performance config
    performance_config = PerformanceConfig(
        metrics_window_size=int(os.environ.get("METRICS_WINDOW_SIZE", "100")),
        min_sample_size=int(os.environ.get("MIN_SAMPLE_SIZE", "5")),
        confidence_threshold=float(os.environ.get("CONFIDENCE_THRESHOLD", "0.6")),
        report_frequency=int(os.environ.get("REPORT_FREQUENCY", "24"))
    )
    
    # Pattern config
    pattern_config = PatternConfig(
        min_pattern_occurrences=int(os.environ.get("MIN_PATTERN_OCCURRENCES", "5")),
        min_pattern_confidence=float(os.environ.get("MIN_PATTERN_CONFIDENCE", "0.7")),
        max_patterns_per_type=int(os.environ.get("MAX_PATTERNS_PER_TYPE", "10")),
        pattern_similarity_threshold=float(os.environ.get("PATTERN_SIMILARITY_THRESHOLD", "0.8"))
    )
    
    # Learning config
    learning_config = LearningConfig(
        reinforcement_learning=rl_config,
        feedback=feedback_config,
        performance=performance_config,
        pattern=pattern_config,
        learning_cycle_interval=int(os.environ.get("LEARNING_CYCLE_INTERVAL", "3600")),
        background_learning=os.environ.get("BACKGROUND_LEARNING", "true").lower() == "true"
    )
    
    # Create learning agent config
    return LearningAgentConfig(
        llm_settings=enhanced_config.llm_settings,
        debug_mode=enhanced_config.debug_mode,
        log_level=enhanced_config.log_level,
        max_history=enhanced_config.max_history,
        memory=enhanced_config.memory,
        learning=learning_config
    )
