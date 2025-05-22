"""
Reinforcement learning implementation for the TAAT AI Agent.

This module provides reinforcement learning mechanisms for the agent to learn
from experience and improve decision-making over time.
"""

import json
import random
import hashlib
from typing import Dict, List, Any, Optional, Tuple

import numpy as np


class ReinforcementLearning:
    """
    Reinforcement learning system for the TAAT AI Agent.
    
    Implements Q-learning for action selection and policy improvement.
    """
    
    def __init__(self, config):
        """
        Initialize the reinforcement learning system.
        
        Args:
            config: Reinforcement learning configuration
        """
        self.config = config
        self.learning_rate = config.learning_rate
        self.discount_factor = config.discount_factor
        self.exploration_rate = config.exploration_rate
        self.min_exploration_rate = config.min_exploration_rate
        self.exploration_decay = config.exploration_decay
        self.reward_scale = config.reward_scale
        
        # State-action values (Q-values)
        self.q_values = {}
        
        # State transition counts
        self.state_transitions = {}
        
        # Action history
        self.action_history = []
    
    def _get_state_key(self, state: Any) -> str:
        """
        Get a unique key for a state.
        
        Args:
            state: State representation
            
        Returns:
            State key
        """
        if isinstance(state, str):
            return f"state:{state}"
        elif isinstance(state, dict):
            # Sort keys for consistent hashing
            state_str = json.dumps(state, sort_keys=True)
            state_hash = hashlib.md5(state_str.encode()).hexdigest()
            return f"state:{state_hash}"
        else:
            # Convert to string and hash
            state_str = str(state)
            state_hash = hashlib.md5(state_str.encode()).hexdigest()
            return f"state:{state_hash}"
    
    def _get_action_key(self, action: Any) -> str:
        """
        Get a unique key for an action.
        
        Args:
            action: Action representation
            
        Returns:
            Action key
        """
        if isinstance(action, str):
            return f"action:{action}"
        elif isinstance(action, dict):
            # Sort keys for consistent hashing
            action_str = json.dumps(action, sort_keys=True)
            action_hash = hashlib.md5(action_str.encode()).hexdigest()
            return f"action:{action_hash}"
        else:
            # Convert to string and hash
            action_str = str(action)
            action_hash = hashlib.md5(action_str.encode()).hexdigest()
            return f"action:{action_hash}"
    
    def _get_state_action_key(self, state_key: str, action_key: str) -> str:
        """
        Get a unique key for a state-action pair.
        
        Args:
            state_key: State key
            action_key: Action key
            
        Returns:
            State-action key
        """
        return f"{state_key}|{action_key}"
    
    async def update_q_value(
        self, state: Any, action: Any, reward: float, next_state: Any
    ) -> float:
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
        # Scale reward
        scaled_reward = reward * self.reward_scale
        
        # Get keys
        state_key = self._get_state_key(state)
        action_key = self._get_action_key(action)
        next_state_key = self._get_state_key(next_state)
        state_action_key = self._get_state_action_key(state_key, action_key)
        
        # Get current Q-value
        current_q = self.q_values.get(state_action_key, 0.0)
        
        # Get max Q-value for next state
        next_q_values = []
        for sa_key, q_value in self.q_values.items():
            if sa_key.startswith(f"{next_state_key}|"):
                next_q_values.append(q_value)
        
        max_next_q = max(next_q_values) if next_q_values else 0.0
        
        # Update Q-value using Q-learning formula
        new_q = current_q + self.learning_rate * (
            scaled_reward + self.discount_factor * max_next_q - current_q
        )
        
        # Store updated Q-value
        self.q_values[state_action_key] = new_q
        
        # Update state transition counts
        transition_key = f"{state_key}|{next_state_key}"
        self.state_transitions[transition_key] = self.state_transitions.get(transition_key, 0) + 1
        
        # Decay exploration rate
        self.exploration_rate = max(
            self.min_exploration_rate,
            self.exploration_rate * self.exploration_decay
        )
        
        return new_q
    
    async def select_action(self, state: Any, possible_actions: List[Any]) -> Any:
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            state: Current state
            possible_actions: List of possible actions
            
        Returns:
            Selected action
        """
        if not possible_actions:
            return None
        
        # Exploration: random action
        if random.random() < self.exploration_rate:
            selected_action = random.choice(possible_actions)
            self.action_history.append({
                "state": state,
                "action": selected_action,
                "type": "exploration",
                "q_value": None
            })
            return selected_action
        
        # Exploitation: best action
        state_key = self._get_state_key(state)
        q_values = []
        
        for action in possible_actions:
            action_key = self._get_action_key(action)
            state_action_key = self._get_state_action_key(state_key, action_key)
            q_value = self.q_values.get(state_action_key, 0.0)
            q_values.append((action, q_value))
        
        # Sort by Q-value (descending)
        q_values.sort(key=lambda x: x[1], reverse=True)
        
        # Return action with highest Q-value
        selected_action = q_values[0][0]
        self.action_history.append({
            "state": state,
            "action": selected_action,
            "type": "exploitation",
            "q_value": q_values[0][1]
        })
        return selected_action
    
    async def get_action_values(self, state: Any) -> Dict[str, float]:
        """
        Get Q-values for all actions in a state.
        
        Args:
            state: State
            
        Returns:
            Dictionary of action keys to Q-values
        """
        state_key = self._get_state_key(state)
        action_values = {}
        
        for sa_key, q_value in self.q_values.items():
            if sa_key.startswith(f"{state_key}|"):
                action_key = sa_key.split("|")[1]
                action_values[action_key] = q_value
        
        return action_values
    
    async def get_state_transition_probabilities(self, state: Any) -> Dict[str, float]:
        """
        Get transition probabilities from a state to other states.
        
        Args:
            state: State
            
        Returns:
            Dictionary of next state keys to transition probabilities
        """
        state_key = self._get_state_key(state)
        transition_counts = {}
        total_transitions = 0
        
        for transition_key, count in self.state_transitions.items():
            if transition_key.startswith(f"{state_key}|"):
                next_state_key = transition_key.split("|")[1]
                transition_counts[next_state_key] = count
                total_transitions += count
        
        # Calculate probabilities
        transition_probs = {}
        for next_state_key, count in transition_counts.items():
            transition_probs[next_state_key] = count / total_transitions if total_transitions > 0 else 0.0
        
        return transition_probs
    
    def save_model(self) -> Dict[str, Any]:
        """
        Save the reinforcement learning model.
        
        Returns:
            Model data
        """
        return {
            "q_values": self.q_values,
            "state_transitions": self.state_transitions,
            "exploration_rate": self.exploration_rate
        }
    
    def load_model(self, model_data: Dict[str, Any]) -> None:
        """
        Load a reinforcement learning model.
        
        Args:
            model_data: Model data
        """
        self.q_values = model_data.get("q_values", {})
        self.state_transitions = model_data.get("state_transitions", {})
        self.exploration_rate = model_data.get("exploration_rate", self.exploration_rate)
