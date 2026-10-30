"""
Unit tests for Contextual Bandit Environment and LinUCB Agent.
"""

import pytest
import numpy as np
from src.contextual_environment import ContextualAdvertisementEnvironment
from src.contextual_agent import LinUCBAgent
from src.simulation import run_contextual_simulation


def test_contextual_environment_init():
    env = ContextualAdvertisementEnvironment(num_ads=4, context_dim=5, seed=42)
    assert env.num_ads == 4
    assert env.context_dim == 5
    assert len(env.ad_names) == 4
    assert len(env.feature_names) == 5
    assert env.theta_matrix.shape == (4, 5)


def test_contextual_context_generation():
    env = ContextualAdvertisementEnvironment(num_ads=4, context_dim=5, seed=42)
    context = env.generate_context()
    assert len(context) == 5
    assert context[0] == 1.0  # Bias term is always 1.0
    # Binary indicators in {0, 1}
    assert context[1] in (0.0, 1.0)
    assert context[2] in (0.0, 1.0)


def test_contextual_probabilities_and_pull():
    env = ContextualAdvertisementEnvironment(num_ads=4, context_dim=5, seed=42)
    context = env.generate_context()
    probs = env.get_ad_click_probabilities(context)
    assert len(probs) == 4
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)

    reward, chosen_prob, regret = env.pull_ad(0, context)
    assert reward in (0, 1)
    assert 0.0 <= chosen_prob <= 1.0
    assert regret >= 0.0  # Regret is non-negative by definition


def test_linucb_agent_init():
    agent = LinUCBAgent(num_ads=4, context_dim=5, alpha=1.0)
    assert agent.num_ads == 4
    assert agent.context_dim == 5
    assert agent.A.shape == (4, 5, 5)
    assert agent.b.shape == (4, 5)

    with pytest.raises(ValueError):
        LinUCBAgent(num_ads=0, context_dim=5)

    with pytest.raises(ValueError):
        LinUCBAgent(num_ads=4, context_dim=0)


def test_linucb_agent_selection_and_update():
    agent = LinUCBAgent(num_ads=3, context_dim=4, alpha=1.0)
    context = np.array([1.0, 0.5, 0.2, 0.8])

    chosen_ad = agent.select_ad(context)
    assert 0 <= chosen_ad < 3

    agent.update(chosen_ad, reward=1, context=context)
    assert agent.counts[chosen_ad] == 1
    assert agent.total_rewards[chosen_ad] == 1
    assert agent.current_round == 1

    # Verify A matrix update
    expected_A_diag = 1.0 + context[0] ** 2
    assert agent.A[chosen_ad, 0, 0] == expected_A_diag


def test_run_contextual_simulation():
    env = ContextualAdvertisementEnvironment(num_ads=3, context_dim=4, seed=42)
    agent = LinUCBAgent(num_ads=3, context_dim=4, alpha=1.0)

    df = run_contextual_simulation(env, agent, num_rounds=100, seed=42)
    assert len(df) == 100
    assert "cumulative_reward" in df.columns
    assert "cumulative_regret" in df.columns
    assert df["cumulative_reward"].iloc[-1] >= 0
    assert df["cumulative_regret"].iloc[-1] >= 0
