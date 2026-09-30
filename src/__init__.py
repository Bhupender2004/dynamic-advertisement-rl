"""
Dynamic Advertisement Selection Using Upper Confidence Bound Reinforcement Learning.

This package provides a clean, modular Multi-Armed Bandit (MAB) simulation framework
implementing the classical UCB1 algorithm for dynamic advertisement selection.
"""

from src.environment import AdvertisementEnvironment
from src.ucb_agent import UCB1Agent
from src.simulation import run_simulation
from src.evaluation import calculate_metrics, generate_plots

__all__ = [
    "AdvertisementEnvironment",
    "UCB1Agent",
    "run_simulation",
    "calculate_metrics",
    "generate_plots",
]
