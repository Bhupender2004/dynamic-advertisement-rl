"""
Upper Confidence Bound (UCB1) Agent for Dynamic Advertisement Selection.

This module implements the classical UCB1 algorithm (Auer et al., 2002) for the
Multi-Armed Bandit problem. It balances exploration of uncertain advertisements
with exploitation of currently known high-performing advertisements.
"""

from typing import Optional
import numpy as np

from src.base_agent import BaseBanditAgent


class UCB1Agent(BaseBanditAgent):
    """
    UCB1 Reinforcement Learning Agent for Multi-Armed Bandits.

    The decision rule computes an Upper Confidence Bound for each advertisement:
        UCB_i(t) = estimated_reward_i + c * sqrt(ln(t) / count_i)

    Where:
        - estimated_reward_i: Sample average reward (empirical CTR) obtained so far.
        - c: Exploration coefficient controlling confidence interval width.
        - t: Current total simulation round number (t >= 1).
        - count_i: Number of times advertisement i has been displayed.

    Attributes:
        num_ads (int): Number of available advertisements (arms).
        c (float): Exploration parameter (default 2.0).
        counts (np.ndarray): Selection counts for each ad [N_i].
        total_rewards (np.ndarray): Total accumulated rewards for each ad [R_i].
        estimated_ctrs (np.ndarray): Empirical CTR estimate for each ad [mu_hat_i].
        current_round (int): Total number of time steps (rounds) completed.
    """

    def __init__(self, num_ads: int, c: float = 2.0, name: str = "UCB1") -> None:
        """
        Initialize the UCB1 Agent.

        Args:
            num_ads: Number of advertisements (must be >= 1).
            c: Exploration parameter (must be > 0.0). Defaults to 2.0.
            name: Human-readable algorithm name.
        """
        if c <= 0.0:
            raise ValueError(f"Exploration coefficient c must be positive, got {c}.")

        super().__init__(num_ads=num_ads, name=name)
        self.c: float = float(c)

    def calculate_ucb(self, ad_index: int, total_rounds: Optional[int] = None) -> float:
        """
        Calculate the Upper Confidence Bound score for a specific advertisement.

        If the advertisement has never been selected (count == 0), its UCB score
        is defined as infinity (np.inf) to guarantee it gets explored at least once.

        Args:
            ad_index: Index of the advertisement.
            total_rounds: Reference round count t. If None, uses max(1, self.current_round).

        Returns:
            float: UCB value for the specified advertisement.
        """
        if self.counts[ad_index] == 0:
            return float(np.inf)

        t = total_rounds if total_rounds is not None else max(1, self.current_round)
        if t <= 0:
            t = 1

        empirical_mean = self.estimated_ctrs[ad_index]
        # Exploration bonus: c * sqrt(ln(t) / N_i)
        exploration_bonus = self.c * np.sqrt(np.log(t) / self.counts[ad_index])

        return float(empirical_mean + exploration_bonus)

    def get_ucb_values(self, total_rounds: Optional[int] = None) -> np.ndarray:
        """
        Compute UCB values across all advertisements for the current round.

        Args:
            total_rounds: Optional round index t.

        Returns:
            np.ndarray: Array of UCB scores for all candidate ads.
        """
        ucb_values = np.zeros(self.num_ads, dtype=np.float64)
        for i in range(self.num_ads):
            ucb_values[i] = self.calculate_ucb(i, total_rounds=total_rounds)
        return ucb_values

    def select_ad(self) -> int:
        """
        Select the advertisement with the highest Upper Confidence Bound.

        Strategy:
        1. Initialization Phase: If any ad has never been selected (count == 0),
           select it immediately to ensure zero-division is avoided.
        2. Optimization Phase: Compute UCB score for all ads and select argmax(UCB).
           Ties are broken consistently.

        Returns:
            int: Selected advertisement index.
        """
        # Step 1: Ensure all arms are pulled at least once during initial exploration
        unselected = np.where(self.counts == 0)[0]
        if len(unselected) > 0:
            return int(unselected[0])

        # Step 2: In round t+1, evaluate UCB using current_round + 1
        t_next = self.current_round + 1
        ucb_values = self.get_ucb_values(total_rounds=t_next)

        # Select arm with maximum UCB value
        selected_ad = int(np.argmax(ucb_values))
        return selected_ad

    def __repr__(self) -> str:
        return (
            f"UCB1Agent(num_ads={self.num_ads}, c={self.c}, "
            f"rounds_completed={self.current_round})"
        )
