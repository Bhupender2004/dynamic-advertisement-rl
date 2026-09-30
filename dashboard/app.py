"""
Streamlit Academic Dashboard for Dynamic Advertisement Selection using UCB.

This interactive web application demonstrates the Multi-Armed Bandit Reinforcement
Learning formulation, Exploration vs. Exploitation trade-off, and UCB1 convergence.
"""

import sys
import os

# Ensure project root is available on sys.path for direct imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.environment import AdvertisementEnvironment
from src.ucb_agent import UCB1Agent
from src.simulation import run_simulation
from src.evaluation import calculate_metrics

# Page configuration
st.set_page_config(
    page_title="Dynamic Ad Selection with UCB",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom styling for clean, academic aesthetic
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748b;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .metric-val {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0f172a;
    }
    .metric-lbl {
        font-size: 0.85rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .formula-box {
        background-color: #f1f5f9;
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header
st.markdown('<div class="main-title">Dynamic Advertisement Selection Using UCB Reinforcement Learning</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Multi-Armed Bandit Simulation & Empirical Convergence Analysis (Review 1 MVP)</div>', unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.header("Simulation Settings")

num_rounds = st.sidebar.slider(
    "Simulation Rounds (t)",
    min_value=500,
    max_value=50000,
    value=10000,
    step=500,
    help="Total number of ad impressions/rounds in the simulation."
)

exploration_c = st.sidebar.slider(
    "Exploration Coefficient (c)",
    min_value=0.1,
    max_value=5.0,
    value=2.0,
    step=0.1,
    help="Controls the weight of the uncertainty bonus in the UCB formula."
)

use_seed = st.sidebar.checkbox("Fixed Random Seed", value=True, help="Ensures exact experimental reproducibility.")
seed_value = st.sidebar.number_input("Seed Value", value=42, step=1, disabled=not use_seed) if use_seed else None

st.sidebar.markdown("---")
st.sidebar.subheader("Advertisement Ground Truth CTRs")
st.sidebar.caption("These hidden probabilities represent real user click behavior. The UCB agent never sees these directly.")

# Preset selector
preset = st.sidebar.selectbox(
    "CTR Configuration Preset",
    ["Standard College Benchmark", "Close Contenders", "High Skew / Clear Winner", "Custom"],
)

if preset == "Standard College Benchmark":
    default_ctrs = [0.05, 0.10, 0.03, 0.15, 0.08]
elif preset == "Close Contenders":
    default_ctrs = [0.12, 0.14, 0.15, 0.13, 0.11]
elif preset == "High Skew / Clear Winner":
    default_ctrs = [0.02, 0.04, 0.03, 0.35, 0.05]
else:
    default_ctrs = [0.05, 0.10, 0.03, 0.15, 0.08]

num_ads = 5
ctrs = []
cols_sb = st.sidebar.columns(2)
for i in range(num_ads):
    with cols_sb[i % 2]:
        val = st.number_input(
            f"Ad {i+1} CTR",
            min_value=0.001,
            max_value=1.0,
            value=float(default_ctrs[i]),
            step=0.01,
            format="%.3f",
            key=f"ctr_ad_{i}",
        )
        ctrs.append(val)

run_button = st.sidebar.button("▶ Run Simulation", type="primary", use_container_width=True)

# Run simulation logic
if "sim_data" not in st.session_state or run_button:
    env = AdvertisementEnvironment(true_ctrs=ctrs, seed=seed_value if use_seed else None)
    agent = UCB1Agent(num_ads=env.num_ads, c=exploration_c)
    df = run_simulation(env=env, agent=agent, num_rounds=num_rounds, seed=seed_value if use_seed else None)
    metrics = calculate_metrics(df, env, agent)
    st.session_state["sim_data"] = {
        "df": df,
        "env": env,
        "agent": agent,
        "metrics": metrics,
        "params": {
            "num_rounds": num_rounds,
            "c": exploration_c,
            "seed": seed_value if use_seed else "Random",
        }
    }

sim = st.session_state["sim_data"]
df = sim["df"]
env = sim["env"]
agent = sim["agent"]
metrics = sim["metrics"]

# Section 1: Overview KPIs
st.subheader("1. Key Performance Indicators")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(
        f"""<div class="metric-card">
            <div class="metric-lbl">Total Impressions</div>
            <div class="metric-val">{metrics['total_rounds']:,}</div>
        </div>""",
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        f"""<div class="metric-card">
            <div class="metric-lbl">Total Clicks (Reward)</div>
            <div class="metric-val">{metrics['total_reward']:,}</div>
        </div>""",
        unsafe_allow_html=True,
    )
with col3:
    st.markdown(
        f"""<div class="metric-card">
            <div class="metric-lbl">Average CTR</div>
            <div class="metric-val">{metrics['average_reward']:.4f}</div>
        </div>""",
        unsafe_allow_html=True,
    )
with col4:
    st.markdown(
        f"""<div class="metric-card">
            <div class="metric-lbl">Optimal Ad (True)</div>
            <div class="metric-val">{metrics['best_ad_name']} ({metrics['max_true_ctr']:.2f})</div>
        </div>""",
        unsafe_allow_html=True,
    )
with col5:
    is_match = metrics['best_ad_name'] == metrics['agent_favorite_name']
    badge = "✅ Correct" if is_match else "⚠️ In Progress"
    st.markdown(
        f"""<div class="metric-card">
            <div class="metric-lbl">Agent Choice</div>
            <div class="metric-val">{metrics['agent_favorite_name']} <span style="font-size:0.9rem;">{badge}</span></div>
        </div>""",
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# Section 2: Advertisement Performance Table
st.subheader("2. Advertisement Performance Breakdown")
st.caption("Comparison between Environment Ground-Truth (hidden during action selection) and UCB Empirical Estimates.")

summary_display = metrics["ad_summary_df"].copy()
summary_display["True CTR"] = summary_display["True CTR"].map("{:.3f}".format)
summary_display["Estimated CTR"] = summary_display["Estimated CTR"].map("{:.4f}".format)
summary_display["Absolute Error"] = summary_display["Absolute Error"].map("{:.4f}".format)
summary_display["Selections"] = summary_display["Selections"].map("{:,}".format)
summary_display["Selection %"] = summary_display["Selection %"].map("{:.2f}%".format)
summary_display["Total Clicks"] = summary_display["Total Clicks"].map("{:,}".format)

st.dataframe(
    summary_display,
    use_container_width=True,
    hide_index=True,
)

# Section 3: Learning Curves (Cumulative Reward & Regret)
st.subheader("3. Learning Performance Dynamics")
col_plot1, col_plot2 = st.columns(2)

rounds = df["round"].values

with col_plot1:
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(rounds, df["cumulative_reward"].values, color="#2563eb", linewidth=2, label="Cumulative Clicks")
    ax1.set_title("Cumulative Reward Over Time", fontweight="bold", fontsize=11)
    ax1.set_xlabel("Round (t)")
    ax1.set_ylabel("Total Clicks")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left")
    st.pyplot(fig1, use_container_width=True)
    plt.close(fig1)

with col_plot2:
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.plot(rounds, df["cumulative_expected_regret"].values, color="#dc2626", linewidth=2, label="Cumulative Expected Regret")
    ax2.set_title("Cumulative Regret Over Time", fontweight="bold", fontsize=11)
    ax2.set_xlabel("Round (t)")
    ax2.set_ylabel("Regret")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper left")
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)

# Section 4 & 5: Ad Selection Frequency & CTR Convergence
st.subheader("4. Exploration vs. Exploitation & CTR Convergence")
col_plot3, col_plot4 = st.columns(2)

true_ctrs = env.get_true_ctrs()
estimated_ctrs = agent.get_estimated_ctrs()
counts = agent.get_selection_counts()
ad_names = env.ad_names
num_ads = env.num_ads

with col_plot3:
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    colors = ["#16a34a" if i == np.argmax(true_ctrs) else "#60a5fa" for i in range(num_ads)]
    bars = ax3.bar(ad_names, counts, color=colors, edgecolor="black", alpha=0.85, width=0.5)
    for bar, count in zip(bars, counts):
        h = bar.get_height()
        ax3.annotate(f"{count:,}", (bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8, fontweight="bold")
    ax3.set_title("Ad Selection Frequency (Exploitation Focus)", fontweight="bold", fontsize=11)
    ax3.set_xlabel("Advertisement")
    ax3.set_ylabel("Impressions (Selections)")
    ax3.grid(True, axis="y", linestyle="--", alpha=0.5)
    st.pyplot(fig3, use_container_width=True)
    plt.close(fig3)

with col_plot4:
    fig4, ax4 = plt.subplots(figsize=(6, 4))
    x_idx = np.arange(num_ads)
    width = 0.35
    ax4.bar(x_idx - width / 2, true_ctrs, width, label="True CTR (Hidden)", color="#4f46e5", alpha=0.85)
    ax4.bar(x_idx + width / 2, estimated_ctrs, width, label="Estimated CTR (UCB)", color="#f59e0b", alpha=0.85)
    ax4.set_title("True CTR vs. Estimated CTR Convergence", fontweight="bold", fontsize=11)
    ax4.set_xlabel("Advertisement")
    ax4.set_ylabel("CTR")
    ax4.set_xticks(x_idx)
    ax4.set_xticklabels(ad_names)
    ax4.legend(loc="upper right")
    ax4.grid(True, axis="y", linestyle="--", alpha=0.5)
    st.pyplot(fig4, use_container_width=True)
    plt.close(fig4)

# Section 6: Mathematical Foundation & UCB Explanation
st.markdown("---")
st.subheader("5. Theoretical Foundation: Upper Confidence Bound (UCB1)")

st.markdown(
    r"""
    The **Upper Confidence Bound (UCB1)** algorithm resolves the exploration-exploitation dilemma by applying the principle of **optimism in the face of uncertainty**.
    
    At round $t$, the score for advertisement $i$ is calculated as:
    """
)

st.latex(r"UCB_i(t) = \hat{\mu}_i + c \sqrt{\frac{\ln(t)}{N_i}}")

st.markdown(
    r"""
    Where:
    - **$\hat{\mu}_i$ (Exploitation Term)**: The empirical average reward obtained from advertisement $i$ ($\hat{\mu}_i = \frac{R_i}{N_i}$).
    - **$c \sqrt{\frac{\ln(t)}{N_i}}$ (Exploration Bonus)**: The upper confidence radius derived from Hoeffding's Inequality representing statistical uncertainty.
    - **$t$**: Total simulation rounds across all advertisements.
    - **$N_i$**: Total selection count for advertisement $i$.
    - **$c$**: Configurable exploration coefficient (balances conservatism vs. risk-seeking).

    **Why logarithmic growth $\ln(t)$?**  
    As total time $t$ progresses, $\ln(t)$ increases slowly, ensuring all arms are checked occasionally, but arms with large $N_i$ experience decaying exploration bonus, causing the agent to asymptotically converge onto the optimal ad with **$O(\ln T)$ logarithmic regret**.
    """
)
