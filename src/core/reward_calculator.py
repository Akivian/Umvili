"""
Reward Calculator Module - Unified Reward Calculation System

This module provides a unified reward calculation system for all learning agents,
eliminating code duplication and ensuring consistent reward shaping across algorithms.

Design Principles:
- Single Responsibility: Only handles reward calculation
- Open-Closed: Easy to extend with new reward components
- DRY: Eliminates duplicate reward calculation code
"""

from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np

from src.core.agent_base import AgentStatus


class RewardComponent(Enum):
    """Reward component types"""
    SUGAR_COLLECTION = "sugar_collection"
    EXPLORATION = "exploration"
    SURVIVAL = "survival"
    EFFICIENCY = "efficiency"
    WEALTH = "wealth"
    MOVEMENT_COST = "movement_cost"
    DEATH_PENALTY = "death_penalty"
    BOUNDARY_PENALTY = "boundary_penalty"


@dataclass
class RewardConfig:
    """Reward calculation configuration"""
    # Sugar collection rewards
    sugar_collection_multiplier: float = 2.0
    sugar_consumption_penalty: float = 0.1
    
    # Exploration rewards
    exploration_reward: float = 0.5
    exploration_enabled: bool = True
    
    # Survival rewards
    survival_reward: float = 0.02
    survival_enabled: bool = True
    
    # Efficiency rewards
    efficiency_multiplier: float = 0.1
    efficiency_min_age: int = 10
    efficiency_enabled: bool = True
    
    # Wealth rewards
    wealth_threshold_1: float = 25.0
    wealth_reward_1: float = 0.1
    wealth_threshold_2: float = 50.0
    wealth_reward_2: float = 0.2
    wealth_enabled: bool = True
    
    # Movement costs
    movement_cost: float = 0.05
    movement_cost_enabled: bool = True
    
    # Death penalty
    death_penalty: float = 5.0
    death_penalty_enabled: bool = True
    
    # Boundary penalty
    boundary_penalty: float = 0.01
    boundary_penalty_enabled: bool = True
    
    # Reward clipping
    min_reward: float = -5.0
    max_reward: float = 5.0
    clip_reward: bool = True


class RewardCalculator:
    """
    Unified reward calculator for learning agents
    
    Provides consistent reward calculation across all learning algorithms,
    with configurable components and weights.
    """
    
    def __init__(self, config: Optional[RewardConfig] = None):
        """
        Initialize reward calculator
        
        Args:
            config: Reward configuration (uses defaults if None)
        """
        self.config = config or RewardConfig()
        self.component_history: Dict[RewardComponent, list] = {
            component: [] for component in RewardComponent
        }
    
    def calculate_reward(
        self,
        prev_sugar: float,
        current_sugar: float,
        prev_position: Tuple[int, int],
        current_position: Tuple[int, int],
        status: AgentStatus,
        age: int,
        total_collected: float,
        visited_positions: set,
        environment_size: int,
        component_breakdown: bool = False
    ) -> Dict[str, Any]:
        """
        Calculate reward for an agent step
        
        Args:
            prev_sugar: Sugar level before action
            current_sugar: Sugar level after action
            prev_position: Position before action
            current_position: Position after action
            status: Current agent status
            age: Agent age
            total_collected: Total sugar collected over lifetime
            visited_positions: Set of visited positions
            environment_size: Size of environment grid
            component_breakdown: Whether to return component breakdown
            
        Returns:
            Dictionary with 'reward' and optionally 'components' breakdown
        """
        reward = 0.0
        components = {}
        
        # 1. Sugar collection reward (most important)
        sugar_change = current_sugar - prev_sugar
        if sugar_change > 0:
            sugar_reward = sugar_change * self.config.sugar_collection_multiplier
            reward += sugar_reward
            components[RewardComponent.SUGAR_COLLECTION] = sugar_reward
        elif sugar_change < 0:
            sugar_penalty = sugar_change * self.config.sugar_consumption_penalty
            reward += sugar_penalty
            components[RewardComponent.SUGAR_COLLECTION] = sugar_penalty
        
        # 2. Exploration reward
        if self.config.exploration_enabled:
            if current_position not in visited_positions:
                reward += self.config.exploration_reward
                components[RewardComponent.EXPLORATION] = self.config.exploration_reward
        
        # 3. Movement cost
        if self.config.movement_cost_enabled:
            if current_position != prev_position:
                reward -= self.config.movement_cost
                components[RewardComponent.MOVEMENT_COST] = -self.config.movement_cost
        
        # 4. Survival reward
        if self.config.survival_enabled:
            if status == AgentStatus.ALIVE:
                reward += self.config.survival_reward
                components[RewardComponent.SURVIVAL] = self.config.survival_reward
        
        # 5. Efficiency reward
        if self.config.efficiency_enabled and age > self.config.efficiency_min_age:
            efficiency = total_collected / age if age > 0 else 0.0
            efficiency_reward = efficiency * self.config.efficiency_multiplier
            reward += efficiency_reward
            components[RewardComponent.EFFICIENCY] = efficiency_reward
        
        # 6. Wealth reward
        if self.config.wealth_enabled:
            if current_sugar > self.config.wealth_threshold_2:
                reward += self.config.wealth_reward_2
                components[RewardComponent.WEALTH] = self.config.wealth_reward_2
            elif current_sugar > self.config.wealth_threshold_1:
                reward += self.config.wealth_reward_1
                components[RewardComponent.WEALTH] = self.config.wealth_reward_1
        
        # 7. Death penalty
        if self.config.death_penalty_enabled:
            if status == AgentStatus.DEAD:
                reward -= self.config.death_penalty
                components[RewardComponent.DEATH_PENALTY] = -self.config.death_penalty
        
        # 8. Boundary penalty
        if self.config.boundary_penalty_enabled:
            x, y = current_position
            if (x == 0 or x == environment_size - 1 or 
                y == 0 or y == environment_size - 1):
                reward -= self.config.boundary_penalty
                components[RewardComponent.BOUNDARY_PENALTY] = -self.config.boundary_penalty
        
        # Clip reward
        if self.config.clip_reward:
            reward = max(self.config.min_reward, min(self.config.max_reward, reward))
        
        # Record component history
        for component, value in components.items():
            self.component_history[component].append(value)
            # Limit history size
            if len(self.component_history[component]) > 1000:
                self.component_history[component].pop(0)
        
        result = {'reward': reward}
        if component_breakdown:
            result['components'] = components
        
        return result
    
    def get_reward_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about reward components
        
        Returns:
            Dictionary with statistics for each component
        """
        stats = {}
        for component, history in self.component_history.items():
            if history:
                stats[component.value] = {
                    'count': len(history),
                    'mean': np.mean(history),
                    'std': np.std(history),
                    'min': np.min(history),
                    'max': np.max(history),
                    'recent_mean': np.mean(history[-100:]) if len(history) >= 100 else np.mean(history)
                }
        return stats
    
    def reset_statistics(self) -> None:
        """Reset reward component history"""
        for component in self.component_history:
            self.component_history[component].clear()


# Convenience function for quick reward calculation
def calculate_agent_reward(
    prev_sugar: float,
    current_sugar: float,
    prev_position: Tuple[int, int],
    current_position: Tuple[int, int],
    status: AgentStatus,
    age: int,
    total_collected: float,
    visited_positions: set,
    environment_size: int,
    config: Optional[RewardConfig] = None
) -> float:
    """
    Quick reward calculation without component breakdown
    
    Args:
        Same as RewardCalculator.calculate_reward
        
    Returns:
        Reward value
    """
    calculator = RewardCalculator(config)
    result = calculator.calculate_reward(
        prev_sugar, current_sugar, prev_position, current_position,
        status, age, total_collected, visited_positions, environment_size
    )
    return result['reward']

