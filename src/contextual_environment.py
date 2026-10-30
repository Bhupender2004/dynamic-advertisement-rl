"""
Contextual Advertisement Environment Module.

This module models an advanced online advertising platform where user click probabilities
depend on both the displayed advertisement AND the user's contextual profile (e.g.,
device type, user age demographic, time of day, category interest).

Theoretical Formulation:
    - User Context: x_t in R^d (normalized feature vector)
    - Hidden Arm Weights: theta_a* in R^d for each ad a = 1 ... K
    - Expected Click Probability: p(click | x_t, a) = sigma(x_t^T theta_a*)
    - Binary Reward: r_t ~ Bernoulli(p(click | x_t, a))
"""

from typing import List, Optional, Tuple, Dict
import numpy as np


class ContextualAdvertisementEnvironment:
    """
    Contextual Multi-Armed Bandit Environment for Personalized Ad Placement.

    Simulates users arriving with distinctive feature vectors x_t.
    Each candidate advertisement has an intrinsic affinity theta_a* toward different
    user features.
    """

    DEFAULT_FEATURE_NAMES = [
        "Bias (Constant)",
        "Mobile User",
        "Age Under 30",
        "Evening / Weekend",
        "Tech & Gaming Affinity",
    ]

    def __init__(
        self,
        num_ads: int = 4,
        context_dim: int = 5,
        ad_names: Optional[List[str]] = None,
        feature_names: Optional[List[str]] = None,
        theta_matrix: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
    ) -> None:
        """
        Initialize Contextual Environment.

        Args:
            num_ads: Number of candidate ads (default: 4).
            context_dim: Dimension d of user context vectors (default: 5).
            ad_names: Human-readable names for ads.
            feature_names: Human-readable names for context features.
            theta_matrix: Optional (num_ads, context_dim) array of true preference weights.
            seed: Random seed for reproducibility.
        """
        self.num_ads = num_ads
        self.context_dim = context_dim
        self.seed = seed
        self._rng = np.random.default_rng(seed)

        if ad_names is None:
            self.ad_names = [
                "Gaming App Ad",
                "Financial Investment Ad",
                "E-Commerce Fashion Ad",
                "Cloud Tech SaaS Ad",
            ][:num_ads]
            if len(self.ad_names) < num_ads:
                for i in range(len(self.ad_names), num_ads):
                    self.ad_names.append(f"Ad {i + 1}")
        else:
            if len(ad_names) != num_ads:
                raise ValueError("Length of ad_names must match num_ads.")
            self.ad_names = list(ad_names)

        if feature_names is None:
            self.feature_names = self.DEFAULT_FEATURE_NAMES[:context_dim]
            if len(self.feature_names) < context_dim:
                for i in range(len(self.feature_names), context_dim):
                    self.feature_names.append(f"Feature {i + 1}")
        else:
            if len(feature_names) != context_dim:
                raise ValueError("Length of feature_names must match context_dim.")
            self.feature_names = list(feature_names)

        # Ground-truth weights theta* for each ad arm
        if theta_matrix is not None:
            if theta_matrix.shape != (num_ads, context_dim):
                raise ValueError(
                    f"theta_matrix shape {theta_matrix.shape} must be ({num_ads}, {context_dim})."
                )
            self.theta_matrix = np.array(theta_matrix, dtype=np.float64)
        else:
            # Domain-specific default weights creating strong personalization signal
            # Row 0: Gaming App -> Likes mobile (+1.8), young (+2.0), tech (+1.5)
            # Row 1: Finance    -> Likes desktop (-1.2 mobile), older (-1.5 young), day time
            # Row 2: Fashion    -> Likes mobile (+1.2), evening (+1.5), general
            # Row 3: Cloud SaaS -> Likes tech (+2.5), desktop (-1.0 mobile)
            base_weights = np.array([
                [-2.0,  1.8,  2.0,  0.8,  1.5],  # Gaming
                [-1.5, -1.2, -1.5, -0.5,  0.5],  # Finance
                [-1.8,  1.2,  0.5,  1.8, -0.5],  # Fashion
                [-2.2, -1.0,  0.2, -0.2,  2.5],  # Cloud SaaS
            ], dtype=np.float64)

            if num_ads == 4 and context_dim == 5:
                self.theta_matrix = base_weights
            else:
                self.theta_matrix = self._rng.normal(0.0, 1.0, size=(num_ads, context_dim))
                # Add baseline negative bias so baseline CTR is ~5-15%
                self.theta_matrix[:, 0] = -2.0

        self.current_context: Optional[np.ndarray] = None

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid function."""
        return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))

    def generate_context(self) -> np.ndarray:
        """
        Sample a realistic user feature vector x_t in R^d.

        Feature 0 is always 1.0 (constant bias term).
        Features 1 to d-1 are binary/continuous user demographic indicators.

        Returns:
            np.ndarray: User context vector of length context_dim.
        """
        context = np.zeros(self.context_dim, dtype=np.float64)
        context[0] = 1.0  # Constant bias

        # User attributes (binary indicators with realistic population distributions)
        if self.context_dim >= 2:
            context[1] = 1.0 if self._rng.random() < 0.65 else 0.0  # 65% mobile
        if self.context_dim >= 3:
            context[2] = 1.0 if self._rng.random() < 0.50 else 0.0  # 50% under 30
        if self.context_dim >= 4:
            context[3] = 1.0 if self._rng.random() < 0.40 else 0.0  # 40% evening
        if self.context_dim >= 5:
            context[4] = 1.0 if self._rng.random() < 0.35 else 0.0  # 35% tech affinity

        # Any extra dimensions sampled uniformly
        for i in range(5, self.context_dim):
            context[i] = float(self._rng.uniform(0.0, 1.0))

        self.current_context = context
        return context

    def get_ad_click_probabilities(self, context: np.ndarray) -> np.ndarray:
        """
        Compute true conditional CTR p(click | context, a) for all candidate ads.

        Args:
            context: Feature vector of shape (context_dim,).

        Returns:
            np.ndarray: Vector of probabilities of length num_ads.
        """
        linear_payoffs = self.theta_matrix @ context
        probs = self._sigmoid(linear_payoffs)
        # Ensure non-trivial probabilities for realistic CTR bounds [0.01, 0.85]
        return np.clip(probs, 0.01, 0.85)

    def pull_ad(self, ad_index: int, context: Optional[np.ndarray] = None) -> Tuple[int, float, float]:
        """
        Simulate user click response to the displayed advertisement given the context.

        Args:
            ad_index: Index of the chosen ad arm.
            context: User context vector (if None, uses current_context).

        Returns:
            Tuple[int, float, float]:
                - reward (0 or 1)
                - true expected CTR for chosen ad
                - instantaneous contextual regret (max_a p_a - p_chosen)
        """
        if context is None:
            if self.current_context is None:
                context = self.generate_context()
            else:
                context = self.current_context

        if not (0 <= ad_index < self.num_ads):
            raise IndexError(f"ad_index {ad_index} out of bounds.")

        probs = self.get_ad_click_probabilities(context)
        chosen_prob = probs[ad_index]
        max_prob = float(np.max(probs))
        regret = float(max_prob - chosen_prob)

        # Sample click ~ Bernoulli(chosen_prob)
        reward = 1 if self._rng.random() < chosen_prob else 0
        return reward, float(chosen_prob), regret

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset environment RNG."""
        if seed is not None:
            self.seed = seed
        self._rng = np.random.default_rng(self.seed)
        self.current_context = None

