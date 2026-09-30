"""Unit tests for UCB1Agent."""

import numpy as np
import pytest
from src.ucb_agent import UCB1Agent


def test_agent_initialization():
    agent = UCB1Agent(num_ads=4, c=2.0)
    assert agent.num_ads == 4
    assert agent.c == 2.0
    assert agent.current_round == 0
    assert np.all(agent.counts == 0)
    assert np.all(agent.total_rewards == 0.0)


def test_invalid_agent_parameters():
    with pytest.raises(ValueError):
        UCB1Agent(num_ads=0)

    with pytest.raises(ValueError):
        UCB1Agent(num_ads=3, c=-1.0)


def test_initial_exploration_phase():
    agent = UCB1Agent(num_ads=3, c=2.0)
    # The first 3 selections should explore every unvisited arm
    selected_1 = agent.select_ad()
    assert selected_1 == 0
    agent.update(0, reward=1)

    selected_2 = agent.select_ad()
    assert selected_2 == 1
    agent.update(1, reward=0)

    selected_3 = agent.select_ad()
    assert selected_3 == 2
    agent.update(2, reward=0)

    assert np.all(agent.counts == 1)


def test_update_statistics():
    agent = UCB1Agent(num_ads=2, c=1.5)
    agent.update(0, reward=1)
    agent.update(0, reward=0)
    agent.update(0, reward=1)

    assert agent.counts[0] == 3
    assert agent.total_rewards[0] == 2
    assert np.isclose(agent.estimated_ctrs[0], 2.0 / 3.0)


def test_ucb_score_calculation():
    agent = UCB1Agent(num_ads=2, c=2.0)
    # Unvisited arm returns infinity
    assert np.isinf(agent.calculate_ucb(0))

    agent.update(0, reward=1)
    # After update at round t=1, count=1, estimated_ctr=1.0, exploration_bonus = 2.0 * sqrt(ln(1)/1) = 0
    ucb_val = agent.calculate_ucb(0, total_rounds=1)
    assert np.isclose(ucb_val, 1.0)

    # For round t=10, count=1, ucb = 1.0 + 2.0 * sqrt(ln(10)/1)
    expected_ucb = 1.0 + 2.0 * np.sqrt(np.log(10) / 1)
    assert np.isclose(agent.calculate_ucb(0, total_rounds=10), expected_ucb)


def test_agent_reset():
    agent = UCB1Agent(num_ads=3)
    agent.update(0, 1)
    agent.update(1, 0)
    agent.reset()
    assert agent.current_round == 0
    assert np.all(agent.counts == 0)
    assert np.all(agent.total_rewards == 0.0)
    assert np.all(agent.estimated_ctrs == 0.0)
