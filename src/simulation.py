"""
Simulation Module for Dynamic Advertisement Selection.

This module coordinates the closed-loop interaction between the simulated
AdvertisementEnvironment and the UCB1Agent across a specified number of rounds.
"""

from typing import Optional, Dict, Any
import numpy as np
import pandas as pd

from src.environment import AdvertisementEnvironment
from src.ucb_agent import UCB1Agent


def run_simulation(
    env: AdvertisementEnvironment,
    agent: UCB1Agent,
    num_rounds: int = 10000,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Execute a full interaction loop between the UCB agent and advertisement environment.

    In each round t = 1 ... num_rounds:
        1. Agent selects an ad based on current UCB estimates.
        2. Environment generates a binary click reward ~ Bernoulli(CTR).
        3. Agent receives feedback and updates its statistics.
        4. Metrics (cumulative reward, expected regret, empirical CTRs) are logged.

    Args:
        env: The simulated AdvertisementEnvironment.
        agent: The UCB1Agent decision maker.
        num_rounds: Total simulation iterations (default: 10,000).
        seed: Optional random seed for reproducible results.

    Returns:
        pd.DataFrame: Complete round-by-round time-series trajectory of the simulation.
    """
    if num_rounds < 1:
        raise ValueError(f"num_rounds must be >= 1, got {num_rounds}.")

    # Reset environment and agent for a clean run
    env.reset(seed=seed)
    agent.reset()

    true_ctrs = env.get_true_ctrs()
    best_ad_idx, best_ad_name, max_true_ctr = env.get_best_ad()

    # Pre-allocate data structures for high performance
    rounds = np.arange(1, num_rounds + 1, dtype=np.int64)
    selected_ads = np.zeros(num_rounds, dtype=np.int64)
    rewards = np.zeros(num_rounds, dtype=np.int64)
    cumulative_rewards = np.zeros(num_rounds, dtype=np.int64)
    inst_expected_regrets = np.zeros(num_rounds, dtype=np.float64)
    cum_expected_regrets = np.zeros(num_rounds, dtype=np.float64)
    inst_actual_regrets = np.zeros(num_rounds, dtype=np.float64)
    cum_actual_regrets = np.zeros(num_rounds, dtype=np.float64)

    # Matrix snapshots for estimated CTRs and UCB values
    estimated_ctrs_history = np.zeros((num_rounds, env.num_ads), dtype=np.float64)
    ucb_values_history = np.zeros((num_rounds, env.num_ads), dtype=np.float64)

    running_reward = 0
    running_exp_regret = 0.0
    running_act_regret = 0.0

    for step in range(num_rounds):
        # 1. Action selection
        chosen_ad = agent.select_ad()

        # 2. Environment interaction (reward sampling)
        reward = env.pull_ad(chosen_ad)

        # 3. Agent learning update
        agent.update(chosen_ad, reward)

        # 4. Metric calculations (for evaluation only)
        running_reward += reward
        expected_loss = max_true_ctr - true_ctrs[chosen_ad]
        running_exp_regret += expected_loss
        actual_loss = max_true_ctr - reward
        running_act_regret += actual_loss

        # Store round data
        selected_ads[step] = chosen_ad
        rewards[step] = reward
        cumulative_rewards[step] = running_reward
        inst_expected_regrets[step] = expected_loss
        cum_expected_regrets[step] = running_exp_regret
        inst_actual_regrets[step] = actual_loss
        cum_actual_regrets[step] = running_act_regret

        estimated_ctrs_history[step] = agent.get_estimated_ctrs()
        ucb_values_history[step] = agent.get_ucb_values(total_rounds=step + 1)

    # Construct clean Pandas DataFrame
    data: Dict[str, Any] = {
        "round": rounds,
        "selected_ad": selected_ads,
        "ad_name": [env.ad_names[i] for i in selected_ads],
        "reward": rewards,
        "cumulative_reward": cumulative_rewards,
        "optimal_expected_reward": np.full(num_rounds, max_true_ctr),
        "instantaneous_regret": inst_actual_regrets,
        "cumulative_regret": cum_actual_regrets,
        "instantaneous_expected_regret": inst_expected_regrets,
        "cumulative_expected_regret": cum_expected_regrets,
    }

    # Add per-ad estimated CTR and UCB value columns
    for i in range(env.num_ads):
        data[f"estimated_ctr_{env.ad_names[i].lower().replace(' ', '_')}"] = (
            estimated_ctrs_history[:, i]
        )
        data[f"ucb_{env.ad_names[i].lower().replace(' ', '_')}"] = (
            ucb_values_history[:, i]
        )

    df = pd.DataFrame(data)
    return df
