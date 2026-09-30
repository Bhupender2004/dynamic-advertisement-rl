"""
Command-Line Entry Point for Dynamic Advertisement Selection using UCB.

This script executes an end-to-end simulation of the UCB1 Multi-Armed Bandit algorithm,
computes performance metrics, saves output artifacts, and generates analytical plots.
"""

import os
from src.environment import AdvertisementEnvironment
from src.ucb_agent import UCB1Agent
from src.simulation import run_simulation
from src.evaluation import calculate_metrics, generate_plots


def main() -> None:
    # 1. Simulation Hyperparameters
    num_rounds = 10000
    exploration_c = 2.0
    random_seed = 42

    # True CTR probabilities (Environment ground-truth, hidden from agent)
    true_ctrs = [0.05, 0.10, 0.03, 0.15, 0.08]
    ad_names = ["Ad 1", "Ad 2", "Ad 3", "Ad 4", "Ad 5"]

    print("=========================================")
    print("Dynamic Advertisement Selection using UCB")
    print("=========================================\n")
    print(f"Number of advertisements: {len(true_ctrs)}")
    print(f"Simulation rounds: {num_rounds}")
    print(f"Exploration coefficient: {exploration_c}")
    print(f"Random seed: {random_seed}\n")

    # 2. Instantiate Environment and Agent
    env = AdvertisementEnvironment(true_ctrs=true_ctrs, ad_names=ad_names, seed=random_seed)
    agent = UCB1Agent(num_ads=env.num_ads, c=exploration_c)

    # 3. Execute Simulation Loop
    print("Running simulation...")
    df = run_simulation(env=env, agent=agent, num_rounds=num_rounds, seed=random_seed)
    print("Simulation completed.\n")

    # 4. Compute Evaluation Metrics
    metrics = calculate_metrics(df, env, agent)

    print("Results:")
    print(f"Total Reward (Clicks): {metrics['total_reward']}")
    print(f"Average Reward (Overall CTR): {metrics['average_reward']:.4f}")
    print(f"Best Advertisement (Ground Truth): {metrics['best_ad_name']} (CTR: {metrics['max_true_ctr']:.2f})")
    print(f"Most Selected Advertisement (Agent): {metrics['agent_favorite_name']}\n")

    print("Advertisement Statistics:")
    print("-" * 75)
    print(f"{'Ad Name':<8} | {'True CTR':<9} | {'Est. CTR':<9} | {'Selections':<10} | {'Selection %':<11} | {'Clicks':<8}")
    print("-" * 75)
    for _, row in metrics["ad_summary_df"].iterrows():
        print(
            f"{row['Advertisement']:<8} | "
            f"{row['True CTR']:<9.2f} | "
            f"{row['Estimated CTR']:<9.4f} | "
            f"{int(row['Selections']):<10} | "
            f"{row['Selection %']:<10.2f}% | "
            f"{int(row['Total Clicks']):<8}"
        )
    print("-" * 75 + "\n")

    # 5. Save Simulation Data
    os.makedirs("data", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    csv_path = os.path.join("data", "simulation_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"Trajectory data saved to: {csv_path}")

    # 6. Generate and Save Visualizations
    plot_paths = generate_plots(df, env, agent, save_dir="results")
    print("Evaluation plots generated successfully in results/:")
    for name, path in plot_paths.items():
        print(f"  - {name}: {path}")

    print("\nResults saved successfully.")


if __name__ == "__main__":
    main()
