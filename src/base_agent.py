"""
Base Bandit Agent Interface for Dynamic Advertisement Selection.

This module provides the abstract base class `BaseBanditAgent` that defines
the common contract for all Multi-Armed Bandit and Reinforcement Learning agents.
"""

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class BaseBanditAgent(ABC):
    """
    Abstract Base Class for Multi-Armed Bandit Agents.

    All bandit agents in this system must implement this interface to ensure
    interchangeability across simulation harnesses, dashboards, and benchmarks.
    """

    def __init__(self, num_ads: int, name: str = "BaseAgent") -> None:
        """
        Initialize the base bandit agent.

        Args:
            num_ads: Number of candidate advertisements (arms), must be >= 1.
            name: Human-readable name of the agent.
        """
        if num_ads < 1:
            raise ValueError(f"num_ads must be at least 1, got {num_ads}.")

        self.num_ads: int = num_ads
        self.name: str = name
        self.counts: np.ndarray = np.zeros(num_ads, dtype=np.int64)
        self.total_rewards: np.ndarray = np.zeros(num_ads, dtype=np.float64)
        self.estimated_ctrs: np.ndarray = np.zeros(num_ads, dtype=np.float64)
        self.current_round: int = 0

    @abstractmethod
    def select_ad(self) -> int:
        """
        Select an advertisement index to display in the current round.

        Returns:
            int: Zero-based index of the chosen advertisement.
        """
        pass

    def update(self, ad_index: int, reward: int) -> None:
        """
        Update the agent's internal state and statistics after receiving feedback.

        Args:
            ad_index: Index of the advertisement that was displayed.
            reward: Observed binary reward (1 for click, 0 for no click).
        """
        if not (0 <= ad_index < self.num_ads):
            raise IndexError(
                f"ad_index {ad_index} is out of bounds for agent with {self.num_ads} ads."
            )
        if reward not in (0, 1):
            raise ValueError(f"reward must be binary (0 or 1), got {reward}.")

        self.current_round += 1
        self.counts[ad_index] += 1
        self.total_rewards[ad_index] += reward
        self.estimated_ctrs[ad_index] = (
            self.total_rewards[ad_index] / self.counts[ad_index]
        )

    def reset(self) -> None:
        """Reset the agent's memory and statistics to initial blank state."""
        self.counts = np.zeros(self.num_ads, dtype=np.int64)
        self.total_rewards = np.zeros(self.num_ads, dtype=np.float64)
        self.estimated_ctrs = np.zeros(self.num_ads, dtype=np.float64)
        self.current_round = 0

    def get_estimated_ctrs(self) -> np.ndarray:
        """Return a copy of the current empirical CTR estimates for each ad."""
        return self.estimated_ctrs.copy()

    def get_selection_counts(self) -> np.ndarray:
        """Return a copy of the selection counts for each ad."""
        return self.counts.copy()

    def get_favorite_ad(self) -> int:
        """Return index of the ad with highest empirical estimated CTR."""
        return int(np.argmax(self.estimated_ctrs))
