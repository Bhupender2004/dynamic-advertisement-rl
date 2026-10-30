"""
Unit tests for multi-agent simulation and offline policy replay evaluator.
"""

import pytest
import os
import pandas as pd
from src.ucb_agent import UCB1Agent
from src.baseline_agents import RandomAgent, ThompsonSamplingAgent, EpsilonGreedyAgent
from src.simulation import run_multi_agent_simulation
from src.offline_evaluator import generate_sample_ad_dataset, evaluate_policy_offline_replay


def test_run_multi_agent_simulation():
    agents = {
        "UCB1": UCB1Agent(num_ads=3, c=1.5),
        "Thompson Sampling": ThompsonSamplingAgent(num_ads=3, seed=42),
        "Random": RandomAgent(num_ads=3, seed=42),
    }

    results = run_multi_agent_simulation(
        agents=agents,
        true_ctrs=[0.05, 0.20, 0.10],
        num_rounds=200,
        seed=42,
    )

    assert set(results.keys()) == {"UCB1", "Thompson Sampling", "Random"}
    for name, df in results.items():
        assert len(df) == 200
        assert "cumulative_reward" in df.columns
        assert "cumulative_expected_regret" in df.columns


def test_generate_sample_ad_dataset(tmp_path):
    out_file = str(tmp_path / "test_clicks.csv")
    df = generate_sample_ad_dataset(output_path=out_file, num_samples=300, num_ads=3, seed=42)

    assert os.path.exists(out_file)
    assert len(df) == 300
    assert "device_mobile" in df.columns
    assert "logged_ad" in df.columns
    assert "click" in df.columns


def test_evaluate_policy_offline_replay(tmp_path):
    out_file = str(tmp_path / "test_clicks.csv")
    df = generate_sample_ad_dataset(output_path=out_file, num_samples=500, num_ads=3, seed=42)

    agent = UCB1Agent(num_ads=3, c=1.5)
    eval_res = evaluate_policy_offline_replay(agent, df, is_contextual=False)

    assert "accepted_rounds" in eval_res
    assert "total_clicks" in eval_res
    assert "effective_ctr" in eval_res
    assert 0 < eval_res["accepted_rounds"] <= 500
    assert 0.0 <= eval_res["effective_ctr"] <= 1.0
