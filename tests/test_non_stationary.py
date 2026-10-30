"""
Unit tests for NonStationaryEnvironment and adaptive Discounted-UCB / SlidingWindow-UCB agents.
"""

import pytest
import numpy as np
from src.non_stationary import NonStationaryEnvironment, DiscountedUCBAgent, SlidingWindowUCBAgent
from src.simulation import run_non_stationary_simulation


def test_non_stationary_environment_shift():
    initial_ctrs = [0.10, 0.20]
    change_points = {50: [0.30, 0.05]}
    env = NonStationaryEnvironment(
        initial_ctrs=initial_ctrs,
        change_points=change_points,
        seed=42,
    )

    # Before shift
    env.step_time(20)
    np.testing.assert_array_equal(env.get_current_ctrs(), [0.10, 0.20])
    best_idx, _, max_ctr = env.get_best_ad()
    assert best_idx == 1
    assert max_ctr == 0.20

    # At change point
    env.step_time(50)
    np.testing.assert_array_equal(env.get_current_ctrs(), [0.30, 0.05])
    best_idx, _, max_ctr = env.get_best_ad()
    assert best_idx == 0
    assert max_ctr == 0.30


def test_discounted_ucb_agent_init():
    agent = DiscountedUCBAgent(num_ads=4, gamma=0.99, c=1.0)
    assert agent.num_ads == 4
    assert agent.gamma == 0.99
    assert agent.c == 1.0

    with pytest.raises(ValueError):
        DiscountedUCBAgent(num_ads=4, gamma=1.5)

    with pytest.raises(ValueError):
        DiscountedUCBAgent(num_ads=4, gamma=-0.1)


def test_discounted_ucb_discounting():
    agent = DiscountedUCBAgent(num_ads=2, gamma=0.9, c=1.0)
    agent.update(0, 1)
    assert agent.discounted_counts[0] == 1.0
    assert agent.discounted_rewards[0] == 1.0

    # Next step: arm 1 gets pulled
    agent.update(1, 0)
    # Arm 0 statistics should have been discounted by gamma (0.9)
    assert pytest.approx(agent.discounted_counts[0]) == 0.9
    assert pytest.approx(agent.discounted_rewards[0]) == 0.9
    assert pytest.approx(agent.discounted_counts[1]) == 1.0


def test_sliding_window_ucb():
    agent = SlidingWindowUCBAgent(num_ads=2, window_size=5, c=1.0)
    for _ in range(10):
        agent.update(0, 1)

    # Total history in deque should not exceed window_size 5
    assert len(agent.history) == 5


def test_non_stationary_simulation():
    env = NonStationaryEnvironment(
        initial_ctrs=[0.05, 0.20],
        change_points={100: [0.25, 0.02]},
        seed=42,
    )
    agents = {
        "D-UCB": DiscountedUCBAgent(num_ads=2, gamma=0.95),
        "SW-UCB": SlidingWindowUCBAgent(num_ads=2, window_size=50),
    }

    results = run_non_stationary_simulation(env, agents, num_rounds=200, seed=42)
    assert "D-UCB" in results
    assert "SW-UCB" in results
    assert len(results["D-UCB"]) == 200
