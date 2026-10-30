"""
Unit tests for baseline bandit agents (Random, EpsilonGreedy, ThompsonSampling).
"""

import pytest
import numpy as np
from src.baseline_agents import RandomAgent, EpsilonGreedyAgent, ThompsonSamplingAgent
from src.environment import AdvertisementEnvironment


def test_random_agent_initialization_and_selection():
    agent = RandomAgent(num_ads=5, seed=42)
    assert agent.num_ads == 5
    assert agent.name == "Random / Static"

    # Selections should be within bounds [0, 4]
    selections = [agent.select_ad() for _ in range(50)]
    assert all(0 <= s < 5 for s in selections)
    # With seed 42 over 50 selections, more than 1 arm should be sampled
    assert len(set(selections)) > 1


def test_random_agent_update():
    agent = RandomAgent(num_ads=3, seed=42)
    agent.update(0, 1)
    agent.update(1, 0)
    assert agent.counts[0] == 1
    assert agent.counts[1] == 1
    assert agent.total_rewards[0] == 1.0
    assert agent.total_rewards[1] == 0.0
    assert agent.current_round == 2


def test_epsilon_greedy_agent_initialization():
    agent = EpsilonGreedyAgent(num_ads=4, epsilon=0.2, decay=False, seed=42)
    assert agent.num_ads == 4
    assert agent.initial_epsilon == 0.2
    assert not agent.decay

    with pytest.raises(ValueError):
        EpsilonGreedyAgent(num_ads=4, epsilon=-0.1)

    with pytest.raises(ValueError):
        EpsilonGreedyAgent(num_ads=4, epsilon=1.5)


def test_epsilon_greedy_decay():
    agent = EpsilonGreedyAgent(num_ads=5, epsilon=0.5, decay=True, seed=42)
    # Before updates
    assert agent.get_current_epsilon() == 0.5

    # Simulate 1000 rounds
    for _ in range(1000):
        ad = agent.select_ad()
        agent.update(ad, 1)

    # Decayed epsilon should be much lower than initial
    assert agent.get_current_epsilon() < 0.5


def test_epsilon_greedy_cold_start():
    agent = EpsilonGreedyAgent(num_ads=3, epsilon=0.1, seed=42)
    # First 3 selections must be 0, 1, 2
    assert agent.select_ad() == 0
    agent.update(0, 0)
    assert agent.select_ad() == 1
    agent.update(1, 0)
    assert agent.select_ad() == 2
    agent.update(2, 0)


def test_thompson_sampling_initialization():
    agent = ThompsonSamplingAgent(num_ads=4, alpha_prior=1.0, beta_prior=1.0, seed=42)
    assert agent.num_ads == 4
    np.testing.assert_array_equal(agent.alphas, np.ones(4))
    np.testing.assert_array_equal(agent.betas, np.ones(4))

    with pytest.raises(ValueError):
        ThompsonSamplingAgent(num_ads=4, alpha_prior=-1.0)


def test_thompson_sampling_update():
    agent = ThompsonSamplingAgent(num_ads=3, seed=42)
    # Ad 0 gets click
    agent.update(0, 1)
    assert agent.alphas[0] == 2.0
    assert agent.betas[0] == 1.0

    # Ad 1 gets no click
    agent.update(1, 0)
    assert agent.alphas[1] == 1.0
    assert agent.betas[1] == 2.0

    # Posterior mean check
    means = agent.get_posterior_means()
    assert means[0] == 2.0 / 3.0
    assert means[1] == 1.0 / 3.0


def test_thompson_sampling_convergence():
    env = AdvertisementEnvironment(true_ctrs=[0.05, 0.50], seed=42)
    agent = ThompsonSamplingAgent(num_ads=2, seed=42)

    for _ in range(500):
        ad = agent.select_ad()
        r = env.pull_ad(ad)
        agent.update(ad, r)

    # Arm 1 (CTR=0.50) should be selected much more than Arm 0 (CTR=0.05)
    assert agent.counts[1] > agent.counts[0]
