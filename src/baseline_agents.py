"""
Baseline Multi-Armed Bandit Agents for Comparative Evaluation.

This module provides standard baseline algorithms for benchmarking against UCB1:
1. RandomAgent: Uniform random exploration (analogous to non-adaptive A/B testing).
2. EpsilonGreedyAgent: Classic epsilon-greedy exploration with optional epsilon decay.
3. ThompsonSamplingAgent: Bayesian probability matching with Beta-Bernoulli conjugate priors.
"""

from typing import Optional
import numpy as np

from src.base_agent import BaseBanditAgent


class RandomAgent(BaseBanditAgent):
    """
    Random Selection Agent.

    Uniformly samples an advertisement at random at every round.
    Serves as an empirical lower bound benchmark demonstrating linear cumulative regret O(T).
    """

    def __init__(
        self,
        num_ads: int,
        seed: Optional[int] = None,
        name: str = "Random / Static",
    ) -> None:
        super().__init__(num_ads=num_ads, name=name)
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def select_ad(self) -> int:
        """Randomly choose an ad from uniform distribution."""
        return int(self._rng.integers(0, self.num_ads))

    def reset(self) -> None:
        super().reset()
        self._rng = np.random.default_rng(self.seed)

    def __repr__(self) -> str:
        return f"RandomAgent(num_ads={self.num_ads})"


class EpsilonGreedyAgent(BaseBanditAgent):
    """
    Epsilon-Greedy Agent with optional decaying exploration rate.

    With probability (1 - epsilon), exploits the current best empirical ad.
    With probability epsilon, explores uniformly at random among all candidate ads.

    If decay=True:
        epsilon(t) = min(1.0, (epsilon_0 * num_ads) / max(1, t))
    """

    def __init__(
        self,
        num_ads: int,
        epsilon: float = 0.1,
        decay: bool = False,
        seed: Optional[int] = None,
        name: Optional[str] = None,
    ) -> None:
        if not (0.0 <= epsilon <= 1.0):
            raise ValueError(f"epsilon must be in [0.0, 1.0], got {epsilon}.")

        self.initial_epsilon: float = float(epsilon)
        self.epsilon: float = float(epsilon)
        self.decay: bool = decay
        self.seed = seed
        self._rng = np.random.default_rng(seed)

        default_name = "Epsilon-Greedy (decay)" if decay else f"Epsilon-Greedy (eps={epsilon})"
        super().__init__(num_ads=num_ads, name=name or default_name)

    def select_ad(self) -> int:
        """
        Select an advertisement using the epsilon-greedy rule.

        Ensures all arms are pulled at least once during initial exploration.
        """
        # Initial cold start
        unselected = np.where(self.counts == 0)[0]
        if len(unselected) > 0:
            return int(unselected[0])

        current_eps = self.get_current_epsilon()

        if self._rng.random() < current_eps:
            # Explore: uniform random choice
            return int(self._rng.integers(0, self.num_ads))
        else:
            # Exploit: choose best empirical arm (break ties randomly)
            max_val = np.max(self.estimated_ctrs)
            best_arms = np.where(self.estimated_ctrs == max_val)[0]
            if len(best_arms) == 1:
                return int(best_arms[0])
            return int(self._rng.choice(best_arms))

    def get_current_epsilon(self) -> float:
        """Compute the current exploration parameter epsilon."""
        if not self.decay or self.current_round == 0:
            return self.initial_epsilon
        # Standard O(K / t) decaying schedule
        decayed = (self.initial_epsilon * self.num_ads) / max(1.0, float(self.current_round))
        return float(min(1.0, max(0.01, decayed)))

    def reset(self) -> None:
        super().reset()
        self.epsilon = self.initial_epsilon
        self._rng = np.random.default_rng(self.seed)

    def __repr__(self) -> str:
        return (
            f"EpsilonGreedyAgent(num_ads={self.num_ads}, epsilon={self.initial_epsilon}, "
            f"decay={self.decay})"
        )


class ThompsonSamplingAgent(BaseBanditAgent):
    """
    Thompson Sampling (Bayesian Posterior Sampling) Agent for Bernoulli Bandits.

    Maintains independent Beta conjugate prior distributions Beta(alpha_i, beta_i)
    over each advertisement's true CTR.

    At each round t:
        1. Sample theta_i ~ Beta(alpha_i, beta_i) for all candidate ads.
        2. Select ad a_t = argmax_i theta_i.
        3. Observe binary reward r_t.
        4. Update posterior:
            alpha_a <- alpha_a + r_t
            beta_a  <- beta_a + (1 - r_t)

    Known to achieve asymptotically optimal logarithmic regret matching the theoretical
    Lai-Robbins lower bound with exceptionally low empirical regret.
    """

    def __init__(
        self,
        num_ads: int,
        alpha_prior: float = 1.0,
        beta_prior: float = 1.0,
        seed: Optional[int] = None,
        name: str = "Thompson Sampling",
    ) -> None:
        if alpha_prior <= 0.0 or beta_prior <= 0.0:
            raise ValueError("Prior hyperparameters alpha and beta must be positive.")

        super().__init__(num_ads=num_ads, name=name)
        self.alpha_prior: float = float(alpha_prior)
        self.beta_prior: float = float(beta_prior)
        self.seed = seed
        self._rng = np.random.default_rng(seed)

        self.alphas: np.ndarray = np.full(num_ads, self.alpha_prior, dtype=np.float64)
        self.betas: np.ndarray = np.full(num_ads, self.beta_prior, dtype=np.float64)

    def select_ad(self) -> int:
        """
        Sample from the posterior Beta distribution for each arm and pick the argmax.
        """
        # Draw one sample from Beta(alpha_i, beta_i) for each arm
        samples = self._rng.beta(self.alphas, self.betas)
        return int(np.argmax(samples))

    def update(self, ad_index: int, reward: int) -> None:
        """
        Update Beta posterior parameters and empirical statistics.
        """
        super().update(ad_index, reward)
        if reward == 1:
            self.alphas[ad_index] += 1.0
        else:
            self.betas[ad_index] += 1.0

    def reset(self) -> None:
        """Reset agent to uninformative or initial prior distributions."""
        super().reset()
        self.alphas = np.full(self.num_ads, self.alpha_prior, dtype=np.float64)
        self.betas = np.full(self.num_ads, self.beta_prior, dtype=np.float64)
        self._rng = np.random.default_rng(self.seed)

    def get_posterior_means(self) -> np.ndarray:
        """Return the expected values of the Beta posteriors: E[CTR] = alpha / (alpha + beta)."""
        return self.alphas / (self.alphas + self.betas)

    def __repr__(self) -> str:
        return (
            f"ThompsonSamplingAgent(num_ads={self.num_ads}, "
            f"prior=Beta({self.alpha_prior}, {self.beta_prior}))"
        )
