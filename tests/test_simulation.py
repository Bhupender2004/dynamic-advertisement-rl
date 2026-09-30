"""Unit tests for simulation and evaluation workflows."""

import numpy as np
import pandas as pd
from src.environment import AdvertisementEnvironment
from src.ucb_agent import UCB1Agent
from src.simulation import run_simulation
from src.evaluation import calculate_metrics


def test_run_simulation_dimensions():
    env = AdvertisementEnvironment(true_ctrs=[0.05, 0.20, 0.10], seed=42)
    agent = UCB1Agent(num_ads=3, c=2.0)
    df = run_simulation(env=env, agent=agent, num_rounds=100, seed=42)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 100
    assert "round" in df.columns
    assert "selected_ad" in df.columns
    assert "reward" in df.columns
    assert "cumulative_reward" in df.columns
    assert "cumulative_expected_regret" in df.columns


def test_cumulative_reward_is_monotonic():
    env = AdvertisementEnvironment(true_ctrs=[0.1, 0.4], seed=123)
    agent = UCB1Agent(num_ads=2, c=1.414)
    df = run_simulation(env=env, agent=agent, num_rounds=200, seed=123)

    cum_rewards = df["cumulative_reward"].values
    assert np.all(np.diff(cum_rewards) >= 0)


def test_simulation_convergence_towards_best_ad():
    # Ad 2 has significantly higher CTR (0.80) compared to Ad 0 (0.05) and Ad 1 (0.10)
    env = AdvertisementEnvironment(true_ctrs=[0.05, 0.10, 0.80], seed=42)
    agent = UCB1Agent(num_ads=3, c=2.0)
    df = run_simulation(env=env, agent=agent, num_rounds=2000, seed=42)
    metrics = calculate_metrics(df, env, agent)

    # Best ad (Ad 3 / index 2) should be selected most frequently
    assert metrics["agent_favorite_index"] == 2
    assert metrics["selection_percentages"][2] > 70.0
