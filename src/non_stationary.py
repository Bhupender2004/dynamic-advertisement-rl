"""
Non-Stationary Multi-Armed Bandit Environment and Adaptive UCB Agents.

This module models real-world advertising dynamics where ad performance decays
over time (ad fatigue, audience saturation, seasonal shifts):
1. NonStationaryEnvironment: Simulates abrupt CTR regime shifts or continuous drift.
2. DiscountedUCBAgent (D-UCB): Exponentially discounts older observations using gamma in (0, 1).
3. SlidingWindowUCBAgent (SW-UCB): Computes empirical CTRs strictly over a sliding window tau.
"""

from typing import List, Optional, Tuple, Dict
from collections import deque
import numpy as np

from src.base_agent import BaseBanditAgent


class NonStationaryEnvironment:
    """
    Advertising Environment with time-varying true Click-Through Rates.

    Supports:
        - 'abrupt': Discontinuous shift in CTRs at configured change points.
        - 'drift': Continuous sinusoidal or linear drift over time.
    """

    def __init__(
        self,
        initial_ctrs: Optional[List[float]] = None,
        ad_names: Optional[List[str]] = None,
        change_points: Optional[Dict[int, List[float]]] = None,
        drift_rate: float = 0.0,
        seed: Optional[int] = None,
    ) -> None:
        """
        Initialize Non-Stationary Environment.

        Args:
            initial_ctrs: Initial CTR distribution [p_1, ..., p_K].
            ad_names: Ad labels.
            change_points: Mapping of round index -> new CTR list.
                           Example: {5000: [0.15, 0.04, 0.02, 0.03, 0.05]}
            drift_rate: Amplitude of continuous periodic drift (if > 0).
            seed: Random seed.
        """
        if initial_ctrs is None:
            # Default: Ad 4 is initial winner (0.15)
            initial_ctrs = [0.05, 0.10, 0.03, 0.15, 0.08]

        self.initial_ctrs = np.array(initial_ctrs, dtype=np.float64)
        self.num_ads = len(self.initial_ctrs)
        self.current_ctrs = self.initial_ctrs.copy()

        if ad_names is None:
            self.ad_names = [f"Ad {i + 1}" for i in range(self.num_ads)]
        else:
            self.ad_names = list(ad_names)

        # Default change point at round 5000: Ad 4 fatigues (0.15 -> 0.02), Ad 2 surges (0.10 -> 0.18)
        if change_points is None:
            self.change_points = {
                5000: [0.05, 0.18, 0.03, 0.02, 0.08]
            }
        else:
            self.change_points = change_points

        self.drift_rate = float(drift_rate)
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self.current_round = 0

    def step_time(self, round_num: int) -> None:
        """Update environmental true CTRs based on the current round number."""
        self.current_round = round_num

        # Check for discrete regime shift
        if round_num in self.change_points:
            new_ctrs = self.change_points[round_num]
            if len(new_ctrs) != self.num_ads:
                raise ValueError("New CTR list length must match num_ads.")
            self.current_ctrs = np.array(new_ctrs, dtype=np.float64)

        # Apply continuous drift if configured
        if self.drift_rate > 0.0:
            sin_offset = self.drift_rate * np.sin(2 * np.pi * round_num / 2000.0)
            self.current_ctrs = np.clip(self.initial_ctrs + sin_offset, 0.01, 0.95)

    def pull_ad(self, ad_index: int, round_num: Optional[int] = None) -> int:
        """
        Pull chosen ad arm and generate binary click reward under current regime.
        """
        if round_num is not None:
            self.step_time(round_num)

        if not (0 <= ad_index < self.num_ads):
            raise IndexError(f"ad_index {ad_index} out of bounds.")

        prob = self.current_ctrs[ad_index]
        return 1 if self._rng.random() < prob else 0

    def get_current_ctrs(self) -> np.ndarray:
        return self.current_ctrs.copy()

    def get_best_ad(self) -> Tuple[int, str, float]:
        best_idx = int(np.argmax(self.current_ctrs))
        return best_idx, self.ad_names[best_idx], float(self.current_ctrs[best_idx])

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self.seed = seed
        self._rng = np.random.default_rng(self.seed)
        self.current_ctrs = self.initial_ctrs.copy()
        self.current_round = 0


class DiscountedUCBAgent(BaseBanditAgent):
    """
    Discounted Upper Confidence Bound (D-UCB) Agent (Garivier & Moulines, 2011).

    In stationary UCB1, all past historical pulls have equal weight 1.
    In D-UCB, observations from tau steps ago are discounted by gamma^tau (gamma in (0, 1)).
    This exponentially decays stale statistics, enabling the agent to rapidly detect
    ad fatigue and shift impressions to newly emerging top-performing ads.
    """

    def __init__(
        self,
        num_ads: int,
        gamma: float = 0.995,
        c: float = 1.0,
        name: str = "Discounted-UCB (D-UCB)",
    ) -> None:
        if not (0.0 < gamma < 1.0):
            raise ValueError(f"Discount factor gamma must be in (0, 1), got {gamma}.")
        if c <= 0.0:
            raise ValueError("Exploration coefficient c must be positive.")

        super().__init__(num_ads=num_ads, name=name)
        self.gamma = float(gamma)
        self.c = float(c)

        # Discounted statistics
        self.discounted_counts = np.zeros(num_ads, dtype=np.float64)
        self.discounted_rewards = np.zeros(num_ads, dtype=np.float64)

    def select_ad(self) -> int:
        """Select ad using the Discounted-UCB formula."""
        # Unselected arms get infinite exploration priority
        unselected = np.where(self.counts == 0)[0]
        if len(unselected) > 0:
            return int(unselected[0])

        total_discounted_n = float(np.sum(self.discounted_counts))
        if total_discounted_n <= 0.0:
            total_discounted_n = 1.0

        ucb_scores = np.zeros(self.num_ads, dtype=np.float64)
        for i in range(self.num_ads):
            n_i = self.discounted_counts[i]
            if n_i <= 1e-6:
                ucb_scores[i] = float(np.inf)
            else:
                emp_mean = self.discounted_rewards[i] / n_i
                bonus = 2.0 * self.c * np.sqrt(np.log(max(1.0, total_discounted_n)) / n_i)
                ucb_scores[i] = emp_mean + bonus

        return int(np.argmax(ucb_scores))

    def update(self, ad_index: int, reward: int) -> None:
        """
        Discount all existing statistics by gamma, then add fresh observation.
        """
        super().update(ad_index, reward)

        # Apply geometric discount factor across all arms
        self.discounted_counts *= self.gamma
        self.discounted_rewards *= self.gamma

        # Increment chosen arm
        self.discounted_counts[ad_index] += 1.0
        self.discounted_rewards[ad_index] += float(reward)

    def reset(self) -> None:
        super().reset()
        self.discounted_counts = np.zeros(self.num_ads, dtype=np.float64)
        self.discounted_rewards = np.zeros(self.num_ads, dtype=np.float64)

    def __repr__(self) -> str:
        return f"DiscountedUCBAgent(num_ads={self.num_ads}, gamma={self.gamma}, c={self.c})"


class SlidingWindowUCBAgent(BaseBanditAgent):
    """
    Sliding-Window UCB (SW-UCB) Agent (Garivier & Moulines, 2011).

    Maintains a rolling FIFO queue of the last W interactions. Older interactions
    past the window boundary W are dropped completely.
    """

    def __init__(
        self,
        num_ads: int,
        window_size: int = 1000,
        c: float = 1.0,
        name: str = "Sliding-Window UCB (SW-UCB)",
    ) -> None:
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}.")
        if c <= 0.0:
            raise ValueError("c must be positive.")

        super().__init__(num_ads=num_ads, name=name)
        self.window_size = int(window_size)
        self.c = float(c)
        self.history: deque = deque(maxlen=window_size)

    def select_ad(self) -> int:
        unselected = np.where(self.counts == 0)[0]
        if len(unselected) > 0:
            return int(unselected[0])

        # Compute counts and rewards within window
        window_counts = np.zeros(self.num_ads, dtype=np.float64)
        window_rewards = np.zeros(self.num_ads, dtype=np.float64)
        for arm, rew in self.history:
            window_counts[arm] += 1.0
            window_rewards[arm] += rew

        total_window_n = float(len(self.history))
        ucb_scores = np.zeros(self.num_ads, dtype=np.float64)

        for i in range(self.num_ads):
            if window_counts[i] == 0:
                ucb_scores[i] = float(np.inf)
            else:
                emp_mean = window_rewards[i] / window_counts[i]
                bonus = self.c * np.sqrt(2.0 * np.log(max(1.0, total_window_n)) / window_counts[i])
                ucb_scores[i] = emp_mean + bonus

        return int(np.argmax(ucb_scores))

    def update(self, ad_index: int, reward: int) -> None:
        super().update(ad_index, reward)
        self.history.append((ad_index, reward))

    def reset(self) -> None:
        super().reset()
        self.history.clear()

    def __repr__(self) -> str:
        return f"SlidingWindowUCBAgent(num_ads={self.num_ads}, window_size={self.window_size})"

