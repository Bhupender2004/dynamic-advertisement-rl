"""
Advertisement Environment Module for Multi-Armed Bandit Simulation.

This module simulates an online advertisement system where each advertisement has
a fixed, hidden Click-Through Rate (CTR). The environment responds to an agent's
action by generating stochastic binary rewards (clicks) sampled from a Bernoulli
distribution parameterized by the chosen ad's true CTR.
"""

from typing import List, Optional, Tuple
import numpy as np


class AdvertisementEnvironment:
    """
    Simulates a Multi-Armed Bandit advertisement platform.

    In this environment, each 'arm' corresponds to an advertisement slot.
    The true Click-Through Rates (CTRs) represent the ground-truth probability
    that a simulated user will click when an advertisement is displayed.

    Attributes:
        true_ctrs (np.ndarray): Underlying hidden click probabilities for each ad.
        ad_names (List[str]): Human-readable identifier labels for each ad.
        num_ads (int): Total number of candidate advertisements available.
    """

    def __init__(
        self,
        true_ctrs: Optional[List[float]] = None,
        ad_names: Optional[List[str]] = None,
        seed: Optional[int] = None,
    ) -> None:
        """
        Initialize the Advertisement Environment.

        Args:
            true_ctrs: List of hidden CTR probabilities (between 0.0 and 1.0).
                       Defaults to [0.05, 0.10, 0.03, 0.15, 0.08].
            ad_names: Optional list of names for advertisements.
            seed: Optional random seed for reproducible reward sampling.
        """
        if true_ctrs is None:
            # Default experimental hidden CTRs as specified in project design
            true_ctrs = [0.05, 0.10, 0.03, 0.15, 0.08]

        self._validate_ctrs(true_ctrs)
        self.true_ctrs: np.ndarray = np.array(true_ctrs, dtype=np.float64)
        self.num_ads: int = len(self.true_ctrs)

        if ad_names is None:
            self.ad_names: List[str] = [f"Ad {i + 1}" for i in range(self.num_ads)]
        else:
            if len(ad_names) != self.num_ads:
                raise ValueError(
                    f"Length of ad_names ({len(ad_names)}) must match true_ctrs ({self.num_ads})."
                )
            self.ad_names = list(ad_names)

        self._seed: Optional[int] = seed
        self._rng: np.random.Generator = np.random.default_rng(seed)

    @staticmethod
    def _validate_ctrs(ctrs: List[float]) -> None:
        """Validate that CTR values are valid probabilities."""
        if not ctrs:
            raise ValueError("true_ctrs cannot be empty.")
        for idx, ctr in enumerate(ctrs):
            if not (0.0 <= ctr <= 1.0):
                raise ValueError(
                    f"CTR for ad index {idx} must be between 0.0 and 1.0, got {ctr}."
                )

    def reset(self, seed: Optional[int] = None) -> None:
        """
        Reset the random generator for a fresh simulation run.

        Args:
            seed: Optional new seed for repeatability.
        """
        if seed is not None:
            self._seed = seed
        self._rng = np.random.default_rng(self._seed)

    def pull_ad(self, ad_index: int) -> int:
        """
        Simulate a user exposure to the selected advertisement.

        The reward is generated using a Bernoulli distribution:
            reward = 1 (click) with probability true_ctr[ad_index]
            reward = 0 (no click) with probability 1 - true_ctr[ad_index]

        Args:
            ad_index: Zero-based index of the advertisement chosen by the agent.

        Returns:
            int: Binary reward (1 for click, 0 for no click).
        """
        if not (0 <= ad_index < self.num_ads):
            raise IndexError(
                f"ad_index {ad_index} out of bounds for environment with {self.num_ads} ads."
            )

        # Sample binary click reward ~ Bernoulli(p = true_ctrs[ad_index])
        click_probability = self.true_ctrs[ad_index]
        reward = int(self._rng.random() < click_probability)
        return reward

    def get_best_ad(self) -> Tuple[int, str, float]:
        """
        Get the optimal advertisement based on true hidden CTR.

        Note: This method is used strictly for evaluation and regret analysis.
        The UCB agent does NOT have access to this information during decision making.

        Returns:
            Tuple[int, str, float]: (best_ad_index, best_ad_name, max_true_ctr)
        """
        best_idx = int(np.argmax(self.true_ctrs))
        return best_idx, self.ad_names[best_idx], float(self.true_ctrs[best_idx])

    def get_true_ctrs(self) -> np.ndarray:
        """
        Return a copy of the true CTR array for benchmarking and plotting.

        Returns:
            np.ndarray: Array of true click-through rates.
        """
        return self.true_ctrs.copy()

    def __repr__(self) -> str:
        return (
            f"AdvertisementEnvironment(num_ads={self.num_ads}, "
            f"true_ctrs={self.true_ctrs.tolist()})"
        )
