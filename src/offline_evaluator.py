"""
Offline Policy Evaluator and Real-World Ad Dataset Replay Engine.

This module implements the provably unbiased offline replay evaluation methodology
introduced by Li et al. (WWW 2010):
    Li, L., Chu, W., Langford, J., & Schapire, R. E. (2010).
    A Contextual-Bandit Approach to Personalized News Headline Recommendation.

Why Offline Evaluation Matters:
    In production digital advertising, deploying an untested RL agent directly to live
    traffic risks substantial revenue loss. Offline policy evaluation provides a safe,
    statistically valid benchmark of new policies on historical logged impression data.
"""

import os
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd


def generate_sample_ad_dataset(
    output_path: str = "data/real_ad_clicks.csv",
    num_samples: int = 10000,
    num_ads: int = 4,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a realistic historical ad impression log dataset.

    Features generated:
        - user_id: Identifier
        - device_type: Mobile (1) or Desktop (0)
        - age_bracket: Young (<30 = 1) or Mature (>=30 = 0)
        - time_of_day: Evening (1) or Day (0)
        - interest_tech: High Tech Interest (1) or General (0)
        - logged_ad: Randomly served ad arm (Uniform Random logging policy)
        - click: Binary user response (Bernoulli with feature-conditioned probabilities)

    Args:
        output_path: Destination CSV file.
        num_samples: Total impression rows (default: 10,000).
        num_ads: Number of candidate ads (default: 4).
        seed: Random seed.

    Returns:
        pd.DataFrame: Generated dataset.
    """
    rng = np.random.default_rng(seed)

    devices = rng.choice([0, 1], size=num_samples, p=[0.35, 0.65])
    ages = rng.choice([0, 1], size=num_samples, p=[0.50, 0.50])
    times = rng.choice([0, 1], size=num_samples, p=[0.60, 0.40])
    tech_interests = rng.choice([0, 1], size=num_samples, p=[0.65, 0.35])

    # Logging policy was uniform random (essential for unbiased replay)
    logged_ads = rng.integers(0, num_ads, size=num_samples)

    # Underlying affinities for realistic click generation
    clicks = np.zeros(num_samples, dtype=np.int64)

    for i in range(num_samples):
        ad = logged_ads[i]
        d, a, t, tech = devices[i], ages[i], times[i], tech_interests[i]

        # Base CTR
        base_logit = -2.2
        if ad == 0:  # Gaming
            logit = base_logit + 1.2 * d + 1.5 * a + 1.0 * tech
        elif ad == 1:  # Finance
            logit = base_logit - 0.8 * d - 1.2 * a + 0.8 * tech
        elif ad == 2:  # Fashion
            logit = base_logit + 1.0 * d + 0.6 * a + 1.4 * t
        else:  # Cloud SaaS
            logit = base_logit - 0.7 * d + 2.0 * tech

        prob = 1.0 / (1.0 + np.exp(-logit))
        clicks[i] = 1 if rng.random() < prob else 0

    df = pd.DataFrame({
        "impression_id": np.arange(1, num_samples + 1),
        "device_mobile": devices,
        "age_under_30": ages,
        "time_evening": times,
        "interest_tech": tech_interests,
        "logged_ad": logged_ads,
        "click": clicks,
    })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


def evaluate_policy_offline_replay(
    agent: Any,
    df_logs: pd.DataFrame,
    context_columns: Optional[List[str]] = None,
    is_contextual: bool = False,
) -> Dict[str, Any]:
    """
    Perform unbiased offline policy evaluation using the Li et al. (2010) Replay Method.

    Replay Mechanism:
        For each log event t:
            1. Agent proposes an action hat{a}_t.
            2. If hat{a}_t matches the logged ad:
               - The event is ACCEPTED.
               - Agent receives the logged reward r_t and updates its policy.
               - Increments accepted trial counter T_accepted.
            3. If hat{a}_t != logged ad:
               - The event is REJECTED (ignored without agent update).

    Returns:
        Dict[str, Any]: Performance metrics over accepted events.
    """
    if context_columns is None:
        context_columns = ["device_mobile", "age_under_30", "time_evening", "interest_tech"]

    agent.reset()

    accepted_rounds = 0
    total_clicks = 0
    running_clicks_history = []
    running_ctr_history = []

    for _, row in df_logs.iterrows():
        logged_ad = int(row["logged_ad"])
        reward = int(row["click"])

        if is_contextual:
            # Construct context vector [1.0 (bias), feat1, feat2, ...]
            features = [1.0] + [float(row[col]) for col in context_columns]
            context_vec = np.array(features, dtype=np.float64)
            proposed_ad = agent.select_ad(context_vec)
        else:
            proposed_ad = agent.select_ad()

        # Replay Match Check
        if proposed_ad == logged_ad:
            accepted_rounds += 1
            total_clicks += reward
            running_clicks_history.append(total_clicks)
            running_ctr_history.append(total_clicks / accepted_rounds)

            # Update agent with matched outcome
            if is_contextual:
                agent.update(proposed_ad, reward, context_vec)
            else:
                agent.update(proposed_ad, reward)

    effective_ctr = (total_clicks / accepted_rounds) if accepted_rounds > 0 else 0.0

    return {
        "accepted_rounds": accepted_rounds,
        "total_clicks": total_clicks,
        "effective_ctr": float(effective_ctr),
        "total_logged_impressions": len(df_logs),
        "replay_acceptance_rate": float(accepted_rounds / len(df_logs)) if len(df_logs) > 0 else 0.0,
        "running_clicks_history": running_clicks_history,
        "running_ctr_history": running_ctr_history,
    }
