"""
Dynamic Advertisement Selection Using Multi-Armed Bandit & Contextual Reinforcement Learning.

This package provides a comprehensive, modular Reinforcement Learning framework for:
- Classical Multi-Armed Bandits (UCB1, Thompson Sampling, Epsilon-Greedy, Random)
- Contextual Personalization Bandits (Disjoint LinUCB)
- Non-Stationary Dynamics & Ad Fatigue (Discounted-UCB, Sliding-Window UCB)
- Unbiased Offline Policy Evaluation (Replay Methodology)
"""

from src.base_agent import BaseBanditAgent
from src.environment import AdvertisementEnvironment
from src.ucb_agent import UCB1Agent
from src.baseline_agents import RandomAgent, EpsilonGreedyAgent, ThompsonSamplingAgent
from src.contextual_environment import ContextualAdvertisementEnvironment
from src.contextual_agent import LinUCBAgent
from src.non_stationary import (
    NonStationaryEnvironment,
    DiscountedUCBAgent,
    SlidingWindowUCBAgent,
)
from src.offline_evaluator import (
    generate_sample_ad_dataset,
    evaluate_policy_offline_replay,
)
from src.simulation import (
    run_simulation,
    run_multi_agent_simulation,
    run_contextual_simulation,
    run_non_stationary_simulation,
)
from src.evaluation import (
    calculate_metrics,
    generate_plots,
    generate_multi_agent_plots,
    generate_contextual_plots,
    generate_non_stationary_plots,
)

__all__ = [
    "BaseBanditAgent",
    "AdvertisementEnvironment",
    "UCB1Agent",
    "RandomAgent",
    "EpsilonGreedyAgent",
    "ThompsonSamplingAgent",
    "ContextualAdvertisementEnvironment",
    "LinUCBAgent",
    "NonStationaryEnvironment",
    "DiscountedUCBAgent",
    "SlidingWindowUCBAgent",
    "generate_sample_ad_dataset",
    "evaluate_policy_offline_replay",
    "run_simulation",
    "run_multi_agent_simulation",
    "run_contextual_simulation",
    "run_non_stationary_simulation",
    "calculate_metrics",
    "generate_plots",
    "generate_multi_agent_plots",
    "generate_contextual_plots",
    "generate_non_stationary_plots",
]
