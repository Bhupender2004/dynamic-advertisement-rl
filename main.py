"""
Master Command-Line Entry Point for Dynamic Advertisement Selection.

This script executes the complete Reinforcement Learning pipeline:
1. Classical UCB1 Multi-Armed Bandit simulation
2. Multi-Algorithm Benchmark Arena (UCB1, Thompson Sampling, ε-Greedy, Random)
3. Contextual Personalization (Disjoint LinUCB with user feature vectors)
4. Non-Stationary Drift & Ad Fatigue (Discounted-UCB adaptation)
5. Offline Policy Replay Evaluation on realistic logged impression data
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.environment import AdvertisementEnvironment
from src.ucb_agent import UCB1Agent
from src.baseline_agents import RandomAgent, EpsilonGreedyAgent, ThompsonSamplingAgent
from src.contextual_environment import ContextualAdvertisementEnvironment
from src.contextual_agent import LinUCBAgent
from src.non_stationary import NonStationaryEnvironment, DiscountedUCBAgent
from src.offline_evaluator import generate_sample_ad_dataset, evaluate_policy_offline_replay
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


def run_ucb1_pipeline(num_rounds: int, seed: int, output_dir: str = "results") -> None:
    print("\n" + "=" * 70)
    print("STAGE 1: Classical UCB1 Multi-Armed Bandit Simulation")
    print("=" * 70)

    true_ctrs = [0.05, 0.10, 0.03, 0.15, 0.08]
    ad_names = ["Ad 1", "Ad 2", "Ad 3", "Ad 4", "Ad 5"]

    env = AdvertisementEnvironment(true_ctrs=true_ctrs, ad_names=ad_names, seed=seed)
    agent = UCB1Agent(num_ads=5, c=2.0)

    print(f"Running {num_rounds:,} rounds with UCB1 (c=2.0)...")
    df = run_simulation(env=env, agent=agent, num_rounds=num_rounds, seed=seed)

    metrics = calculate_metrics(df, env, agent)
    print(f"\n[UCB1 Results]")
    print(f"Total Clicks Harvested: {metrics['total_reward']:,}")
    print(f"Average CTR:            {metrics['average_reward']:.4f}")
    print(f"Theoretical Regret:     {metrics['total_expected_regret']:.2f}")
    print(f"Optimal Ad (Ground Truth): {metrics['best_ad_name']} (CTR: {metrics['max_true_ctr']:.2f})")
    print(f"Winner Identified:         {metrics['agent_favorite_name']}\n")

    print(metrics["ad_summary_df"].to_string(index=False))

    # Save trajectories and plots
    os.makedirs("data", exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv("data/simulation_results.csv", index=False)
    plots = generate_plots(df, env, agent, save_dir=output_dir)
    print(f"\nSaved UCB1 plots to {output_dir}/ ({len(plots)} figures).")


def run_multi_agent_pipeline(num_rounds: int, seed: int, output_dir: str = "results") -> None:
    print("\n" + "=" * 70)
    print("STAGE 2: Multi-Algorithm Benchmark Arena")
    print("=" * 70)

    true_ctrs = [0.05, 0.10, 0.03, 0.15, 0.08]

    agents = {
        "UCB1": UCB1Agent(num_ads=5, c=2.0),
        "Thompson Sampling": ThompsonSamplingAgent(num_ads=5, seed=seed),
        "Epsilon-Greedy (decay)": EpsilonGreedyAgent(num_ads=5, epsilon=0.5, decay=True, seed=seed),
        "Epsilon-Greedy (eps=0.1)": EpsilonGreedyAgent(num_ads=5, epsilon=0.1, decay=False, seed=seed),
        "Random / Static": RandomAgent(num_ads=5, seed=seed),
    }

    print(f"Benchmarking {len(agents)} algorithms over {num_rounds:,} rounds...")
    results = run_multi_agent_simulation(agents=agents, true_ctrs=true_ctrs, num_rounds=num_rounds, seed=seed)

    leaderboard = []
    for name, df in results.items():
        clicks = int(df["cumulative_reward"].iloc[-1])
        regret = float(df["cumulative_expected_regret"].iloc[-1])
        leaderboard.append({
            "Algorithm": name,
            "Total Clicks": clicks,
            "System CTR": f"{clicks / len(df):.4f}",
            "Cumulative Regret": f"{regret:.2f}",
        })

    lb_df = pd.DataFrame(leaderboard).sort_values(by="Total Clicks", ascending=False)
    print("\n[Algorithm Benchmark Leaderboard]")
    print(lb_df.to_string(index=False))

    lb_df.to_csv("data/multi_agent_leaderboard.csv", index=False)
    plots = generate_multi_agent_plots(results, save_dir=output_dir)
    print(f"\nSaved Multi-Agent plots to {output_dir}/ ({len(plots)} figures).")


def run_contextual_pipeline(num_rounds: int, seed: int, output_dir: str = "results") -> None:
    print("\n" + "=" * 70)
    print("STAGE 3: Contextual Bandits (Disjoint LinUCB Personalization)")
    print("=" * 70)

    c_env = ContextualAdvertisementEnvironment(num_ads=4, context_dim=5, seed=seed)
    c_agent = LinUCBAgent(num_ads=4, context_dim=5, alpha=1.0)

    print(f"Simulating {num_rounds:,} personalized user visits with LinUCB...")
    df_contextual = run_contextual_simulation(c_env, c_agent, num_rounds=num_rounds, seed=seed)

    clicks = int(df_contextual["cumulative_reward"].iloc[-1])
    regret = float(df_contextual["cumulative_regret"].iloc[-1])

    print(f"\n[LinUCB Results]")
    print(f"Total Clicks Harvested:     {clicks:,}")
    print(f"Personalized System CTR:    {clicks / num_rounds:.4f}")
    print(f"Cumulative Contextual Loss: {regret:.2f}")

    df_contextual.to_csv("data/contextual_results.csv", index=False)
    plots = generate_contextual_plots(df_contextual, save_dir=output_dir)
    print(f"\nSaved Contextual plots to {output_dir}/.")


def run_non_stationary_pipeline(num_rounds: int, seed: int, output_dir: str = "results") -> None:
    print("\n" + "=" * 70)
    print("STAGE 4: Non-Stationary Dynamics & Ad Fatigue Adaptation")
    print("=" * 70)

    change_point = num_rounds // 2
    ns_env = NonStationaryEnvironment(
        initial_ctrs=[0.05, 0.10, 0.03, 0.15, 0.08],
        change_points={change_point: [0.05, 0.18, 0.03, 0.02, 0.08]},
        seed=seed,
    )

    from src.non_stationary import SlidingWindowUCBAgent

    ns_agents = {
        "UCB1 (Stationary)": UCB1Agent(num_ads=5, c=1.0),
        "Discounted-UCB (D-UCB)": DiscountedUCBAgent(num_ads=5, gamma=0.998, c=0.2),
        "Sliding-Window UCB (SW-UCB)": SlidingWindowUCBAgent(num_ads=5, window_size=1500, c=0.2),
    }

    print(f"Simulating regime shift at round {change_point:,} (Ad 4 collapses 15% -> 2%, Ad 2 surges 10% -> 18%)...")
    ns_results = run_non_stationary_simulation(ns_env, ns_agents, num_rounds=num_rounds, seed=seed)

    u_reg = float(ns_results["UCB1 (Stationary)"]["cumulative_regret"].iloc[-1])
    d_reg = float(ns_results["Discounted-UCB (D-UCB)"]["cumulative_regret"].iloc[-1])
    sw_reg = float(ns_results["Sliding-Window UCB (SW-UCB)"]["cumulative_regret"].iloc[-1])

    print(f"\n[Non-Stationary Results]")
    print(f"Stationary UCB1 Final Regret:      {u_reg:.2f}")
    print(f"Discounted-UCB (D-UCB) Regret:     {d_reg:.2f}")
    print(f"Sliding-Window UCB (SW-UCB) Regret:{sw_reg:.2f} ({(1.0 - sw_reg / u_reg) * 100:.1f}% regret reduction!)")

    plots = generate_non_stationary_plots(ns_results, change_point=change_point, save_dir=output_dir)
    print(f"\nSaved Non-Stationary plots to {output_dir}/.")


def run_offline_eval_pipeline(num_samples: int = 10000, seed: int = 42) -> None:
    print("\n" + "=" * 70)
    print("STAGE 5: Unbiased Offline Policy Evaluation (Replay Method)")
    print("=" * 70)

    dataset_path = "data/real_ad_clicks.csv"
    if not os.path.exists(dataset_path):
        print(f"Generating realistic ad impression log dataset ({num_samples:,} rows)...")
        generate_sample_ad_dataset(output_path=dataset_path, num_samples=num_samples, seed=seed)

    df_logs = pd.read_csv(dataset_path)
    print(f"Evaluating policies against {len(df_logs):,} logged historical impression records...")

    eval_ucb = evaluate_policy_offline_replay(UCB1Agent(num_ads=4, c=1.5), df_logs, is_contextual=False)
    eval_ts = evaluate_policy_offline_replay(ThompsonSamplingAgent(num_ads=4, seed=seed), df_logs, is_contextual=False)
    eval_linucb = evaluate_policy_offline_replay(LinUCBAgent(num_ads=4, context_dim=5, alpha=1.0), df_logs, is_contextual=True)

    summary = pd.DataFrame([
        {"Policy": "UCB1", "Accepted Impressions": f"{eval_ucb['accepted_rounds']:,}", "Clicks": eval_ucb['total_clicks'], "Effective CTR": f"{eval_ucb['effective_ctr']:.4f}"},
        {"Policy": "Thompson Sampling", "Accepted Impressions": f"{eval_ts['accepted_rounds']:,}", "Clicks": eval_ts['total_clicks'], "Effective CTR": f"{eval_ts['effective_ctr']:.4f}"},
        {"Policy": "Contextual LinUCB", "Accepted Impressions": f"{eval_linucb['accepted_rounds']:,}", "Clicks": eval_linucb['total_clicks'], "Effective CTR": f"{eval_linucb['effective_ctr']:.4f}"},
    ])

    print("\n[Offline Replay Evaluation Summary]")
    print(summary.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Dynamic Advertisement Selection using Multi-Armed Bandits & RL.")
    parser.add_argument("--mode", type=str, choices=["all", "ucb", "benchmark", "contextual", "nonstationary", "offline"], default="all")
    parser.add_argument("--rounds", type=int, default=10000, help="Simulation rounds (default: 10,000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("DYNAMIC ADVERTISEMENT SELECTION USING REINFORCEMENT LEARNING")
    print("Master Capstone Benchmark Execution")
    print("=" * 70)

    if args.mode in ("all", "ucb"):
        run_ucb1_pipeline(num_rounds=args.rounds, seed=args.seed)

    if args.mode in ("all", "benchmark"):
        run_multi_agent_pipeline(num_rounds=args.rounds, seed=args.seed)

    if args.mode in ("all", "contextual"):
        run_contextual_pipeline(num_rounds=args.rounds, seed=args.seed)

    if args.mode in ("all", "nonstationary"):
        run_non_stationary_pipeline(num_rounds=args.rounds, seed=args.seed)

    if args.mode in ("all", "offline"):
        run_offline_eval_pipeline(num_samples=args.rounds, seed=args.seed)

    print("\n" + "=" * 70)
    print("ALL PIPELINE BENCHMARKS EXECUTED SUCCESSFULLY!")
    print("Artifacts generated in data/ and results/.")
    print("Launch dashboard: streamlit run dashboard/app.py")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
