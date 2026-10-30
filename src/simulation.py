"""
Simulation Module for Dynamic Advertisement Selection.

Coordinates the closed-loop interaction loops for:
1. Single Agent Stationary Simulation (run_simulation)
2. Multi-Agent Benchmark Arena (run_multi_agent_simulation)
3. Contextual Bandit Personalization Loop (run_contextual_simulation)
4. Non-Stationary Drift & Regime Shift Loop (run_non_stationary_simulation)
"""

from typing import Optional, Dict, Any, List
import numpy as np
import pandas as pd

from src.environment import AdvertisementEnvironment
from src.ucb_agent import UCB1Agent
from src.contextual_environment import ContextualAdvertisementEnvironment
from src.contextual_agent import LinUCBAgent
from src.non_stationary import NonStationaryEnvironment, DiscountedUCBAgent


def run_simulation(
    env: AdvertisementEnvironment,
    agent: Any,
    num_rounds: int = 10000,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Execute a full interaction loop between a bandit agent and advertisement environment.

    In each round t = 1 ... num_rounds:
        1. Agent selects an ad based on its policy.
        2. Environment generates a binary click reward ~ Bernoulli(CTR).
        3. Agent receives feedback and updates its statistics.
        4. Metrics (cumulative reward, expected regret, empirical CTRs) are logged.

    Args:
        env: The simulated AdvertisementEnvironment.
        agent: The decision-making bandit agent.
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

    # Matrix snapshots for estimated CTRs and UCB values (if supported)
    estimated_ctrs_history = np.zeros((num_rounds, env.num_ads), dtype=np.float64)
    has_ucb_values = hasattr(agent, "get_ucb_values")
    ucb_values_history = np.zeros((num_rounds, env.num_ads), dtype=np.float64) if has_ucb_values else None

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
        if has_ucb_values and ucb_values_history is not None:
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
        clean_name = env.ad_names[i].lower().replace(" ", "_")
        data[f"estimated_ctr_{clean_name}"] = estimated_ctrs_history[:, i]
        if has_ucb_values and ucb_values_history is not None:
            data[f"ucb_{clean_name}"] = ucb_values_history[:, i]

    df = pd.DataFrame(data)
    return df


def run_multi_agent_simulation(
    agents: Dict[str, Any],
    true_ctrs: Optional[List[float]] = None,
    num_rounds: int = 10000,
    seed: int = 42,
) -> Dict[str, pd.DataFrame]:
    """
    Run multiple bandit agents simultaneously against an identical environment sequence.

    Args:
        agents: Dict of algorithm name -> agent instance.
        true_ctrs: Hidden ad CTR distribution.
        num_rounds: Simulation rounds.
        seed: Random seed.

    Returns:
        Dict[str, pd.DataFrame]: Mapping of agent name to trajectory DataFrame.
    """
    results = {}
    for name, agent in agents.items():
        # Instantiate fresh environment with identical seed for fair comparison
        env = AdvertisementEnvironment(true_ctrs=true_ctrs, seed=seed)
        df = run_simulation(env=env, agent=agent, num_rounds=num_rounds, seed=seed)
        results[name] = df
    return results


def run_contextual_simulation(
    env: ContextualAdvertisementEnvironment,
    agent: LinUCBAgent,
    num_rounds: int = 10000,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Execute contextual interaction loop where user feature vectors precede action selection.

    Args:
        env: ContextualAdvertisementEnvironment.
        agent: LinUCBAgent.
        num_rounds: Number of rounds.
        seed: Random seed.

    Returns:
        pd.DataFrame: Round-by-round contextual trajectory.
    """
    env.reset(seed=seed)
    agent.reset()

    rounds = np.arange(1, num_rounds + 1, dtype=np.int64)
    selected_ads = np.zeros(num_rounds, dtype=np.int64)
    rewards = np.zeros(num_rounds, dtype=np.int64)
    cumulative_rewards = np.zeros(num_rounds, dtype=np.int64)
    instant_regrets = np.zeros(num_rounds, dtype=np.float64)
    cumulative_regrets = np.zeros(num_rounds, dtype=np.float64)
    true_click_probs = np.zeros(num_rounds, dtype=np.float64)

    running_reward = 0
    running_regret = 0.0

    for step in range(num_rounds):
        # 1. User arrives with context vector
        context = env.generate_context()

        # 2. Agent recommends ad conditioned on context
        chosen_ad = agent.select_ad(context)

        # 3. Environment produces feedback
        reward, true_prob, regret = env.pull_ad(chosen_ad, context)

        # 4. Agent updates ridge regression weights
        agent.update(chosen_ad, reward, context)

        running_reward += reward
        running_regret += regret

        selected_ads[step] = chosen_ad
        rewards[step] = reward
        cumulative_rewards[step] = running_reward
        instant_regrets[step] = regret
        cumulative_regrets[step] = running_regret
        true_click_probs[step] = true_prob

    df = pd.DataFrame({
        "round": rounds,
        "selected_ad": selected_ads,
        "ad_name": [env.ad_names[i] for i in selected_ads],
        "reward": rewards,
        "cumulative_reward": cumulative_rewards,
        "instantaneous_regret": instant_regrets,
        "cumulative_regret": cumulative_regrets,
        "expected_ctr": true_click_probs,
    })
    return df


def run_non_stationary_simulation(
    env: NonStationaryEnvironment,
    agents: Dict[str, Any],
    num_rounds: int = 10000,
    seed: Optional[int] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Benchmark stationary agents vs. adaptive agents on a time-varying environment.
    """
    results = {}
    for name, agent in agents.items():
        env.reset(seed=seed)
        agent.reset()

        rounds = np.arange(1, num_rounds + 1, dtype=np.int64)
        selected_ads = np.zeros(num_rounds, dtype=np.int64)
        rewards = np.zeros(num_rounds, dtype=np.int64)
        cum_rewards = np.zeros(num_rounds, dtype=np.int64)
        cum_regrets = np.zeros(num_rounds, dtype=np.float64)

        running_rew = 0
        running_reg = 0.0

        for step in range(num_rounds):
            round_idx = step + 1
            env.step_time(round_idx)

            chosen = agent.select_ad()
            reward = env.pull_ad(chosen)
            agent.update(chosen, reward)

            current_ctrs = env.get_current_ctrs()
            max_ctr = np.max(current_ctrs)
            regret = max_ctr - current_ctrs[chosen]

            running_rew += reward
            running_reg += regret

            selected_ads[step] = chosen
            rewards[step] = reward
            cum_rewards[step] = running_rew
            cum_regrets[step] = running_reg

        results[name] = pd.DataFrame({
            "round": rounds,
            "selected_ad": selected_ads,
            "reward": rewards,
            "cumulative_reward": cum_rewards,
            "cumulative_regret": cum_regrets,
        })

    return results
