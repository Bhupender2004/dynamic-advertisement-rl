"""
Contextual Bandit Agent: Disjoint LinUCB for Personalized Advertisement Selection.

Reference:
    Li, L., Chu, W., Langford, J., & Schapire, R. E. (2010).
    A Contextual-Bandit Approach to Personalized News Headline Recommendation.
    In Proceedings of the 19th International Conference on World Wide Web (WWW '10), pp. 661–670.

Mathematical Formulation:
    For each candidate advertisement arm a in {1 ... K}:
        - Maintain design matrix covariance A_a = D_a^T D_a + I_d (initialized as I_d)
        - Maintain response vector b_a = D_a^T c_a (initialized as 0_d)
        - Ridge regression estimator: hat{theta}_a = A_a^{-1} b_a
        - Confidence radius: alpha * sqrt(x_t^T A_a^{-1} x_t)
        - Upper Confidence Bound score:
            p_{t, a} = hat{theta}_a^T x_t + alpha * sqrt(x_t^T A_a^{-1} x_t)
        - Decision rule: a_t = argmax_a p_{t, a}
        - Online Update:
            A_{a_t} <- A_{a_t} + x_t x_t^T
            b_{a_t} <- b_{a_t} + r_t x_t
"""

from typing import Optional, List
import numpy as np


class LinUCBAgent:
    """
    Disjoint Linear Upper Confidence Bound (LinUCB) Agent.

    Enables personalized recommendation by associating an independent linear
    model with each candidate advertisement arm.
    """

    def __init__(
        self,
        num_ads: int,
        context_dim: int,
        alpha: float = 1.0,
        name: str = "LinUCB",
    ) -> None:
        """
        Initialize the LinUCB Agent.

        Args:
            num_ads: Number of candidate advertisements (arms).
            context_dim: Dimension d of user feature vectors.
            alpha: Exploration coefficient scaling the confidence bound (alpha > 0).
            name: Human-readable name.
        """
        if num_ads < 1:
            raise ValueError(f"num_ads must be >= 1, got {num_ads}.")
        if context_dim < 1:
            raise ValueError(f"context_dim must be >= 1, got {context_dim}.")
        if alpha <= 0.0:
            raise ValueError(f"alpha must be positive, got {alpha}.")

        self.num_ads = num_ads
        self.context_dim = context_dim
        self.alpha = float(alpha)
        self.name = name

        # A_a: (num_ads, d, d) initialized to Identity
        self.A = np.zeros((num_ads, context_dim, context_dim), dtype=np.float64)
        for a in range(num_ads):
            self.A[a] = np.identity(context_dim, dtype=np.float64)

        # b_a: (num_ads, d) initialized to Zeros
        self.b = np.zeros((num_ads, context_dim), dtype=np.float64)

        # Selection and click counts
        self.counts = np.zeros(num_ads, dtype=np.int64)
        self.total_rewards = np.zeros(num_ads, dtype=np.int64)
        self.current_round = 0

    def select_ad(self, context: np.ndarray) -> int:
        """
        Select the optimal advertisement for the given user context vector.

        Args:
            context: User feature vector x_t of shape (context_dim,).

        Returns:
            int: Selected advertisement arm index.
        """
        if len(context) != self.context_dim:
            raise ValueError(
                f"Expected context of length {self.context_dim}, got {len(context)}."
            )

        x = context.reshape(-1, 1)  # (d, 1)
        scores = np.zeros(self.num_ads, dtype=np.float64)

        for a in range(self.num_ads):
            # A_inv = np.linalg.inv(self.A[a])
            A_inv = np.linalg.inv(self.A[a])
            theta_hat = A_inv @ self.b[a].reshape(-1, 1)  # (d, 1)

            # Exploitation mean: theta_hat^T x
            mean_est = float((theta_hat.T @ x)[0, 0])

            # Exploration bonus: alpha * sqrt(x^T A_inv x)
            variance = float((x.T @ A_inv @ x)[0, 0])
            # Ensure non-negative due to numerical precision
            bonus = self.alpha * np.sqrt(max(0.0, variance))

            scores[a] = mean_est + bonus

        # Break ties randomly or deterministically
        return int(np.argmax(scores))

    def update(self, ad_index: int, reward: int, context: np.ndarray) -> None:
        """
        Update the ridge regression model for the chosen advertisement.

        Args:
            ad_index: Index of the served advertisement.
            reward: Binary click outcome (0 or 1).
            context: The context vector x_t for this interaction.
        """
        if not (0 <= ad_index < self.num_ads):
            raise IndexError(f"ad_index {ad_index} out of bounds.")
        if reward not in (0, 1):
            raise ValueError("Reward must be 0 or 1.")

        x = context.reshape(-1, 1)  # (d, 1)

        # Online Sherman-Morrison rank-1 update
        self.A[ad_index] += x @ x.T
        self.b[ad_index] += float(reward) * context

        self.counts[ad_index] += 1
        self.total_rewards[ad_index] += reward
        self.current_round += 1

    def reset(self) -> None:
        """Reset internal parameters to uninformative initial state."""
        for a in range(self.num_ads):
            self.A[a] = np.identity(self.context_dim, dtype=np.float64)
        self.b = np.zeros((self.num_ads, self.context_dim), dtype=np.float64)
        self.counts = np.zeros(self.num_ads, dtype=np.int64)
        self.total_rewards = np.zeros(self.num_ads, dtype=np.int64)
        self.current_round = 0

    def get_estimated_thetas(self) -> np.ndarray:
        """
        Compute and return the learned weight vectors hat{theta}_a for each ad.

        Returns:
            np.ndarray: Matrix of shape (num_ads, context_dim).
        """
        thetas = np.zeros((self.num_ads, self.context_dim), dtype=np.float64)
        for a in range(self.num_ads):
            thetas[a] = np.linalg.solve(self.A[a], self.b[a])
        return thetas

    def __repr__(self) -> str:
        return (
            f"LinUCBAgent(num_ads={self.num_ads}, context_dim={self.context_dim}, "
            f"alpha={self.alpha})"
        )
