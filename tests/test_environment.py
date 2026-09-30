"""Unit tests for AdvertisementEnvironment."""

import pytest
import numpy as np
from src.environment import AdvertisementEnvironment


def test_environment_initialization():
    env = AdvertisementEnvironment(true_ctrs=[0.1, 0.2, 0.3])
    assert env.num_ads == 3
    assert len(env.ad_names) == 3
    assert np.allclose(env.get_true_ctrs(), [0.1, 0.2, 0.3])


def test_invalid_ctrs():
    with pytest.raises(ValueError):
        AdvertisementEnvironment(true_ctrs=[])

    with pytest.raises(ValueError):
        AdvertisementEnvironment(true_ctrs=[-0.1, 0.5])

    with pytest.raises(ValueError):
        AdvertisementEnvironment(true_ctrs=[0.2, 1.5])


def test_pull_ad_returns_binary_reward():
    env = AdvertisementEnvironment(true_ctrs=[0.3, 0.7], seed=42)
    rewards = [env.pull_ad(0) for _ in range(100)]
    assert all(r in [0, 1] for r in rewards)


def test_best_ad_identification():
    env = AdvertisementEnvironment(true_ctrs=[0.05, 0.25, 0.10])
    best_idx, best_name, max_ctr = env.get_best_ad()
    assert best_idx == 1
    assert best_name == "Ad 2"
    assert max_ctr == 0.25


def test_out_of_bounds_pull():
    env = AdvertisementEnvironment(true_ctrs=[0.1, 0.2])
    with pytest.raises(IndexError):
        env.pull_ad(5)
