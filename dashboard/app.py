"""
Streamlit Academic Research Dashboard for Dynamic Advertisement Selection.

An interactive Multi-Armed Bandit & Contextual Reinforcement Learning platform:
- Explicit Configuration -> Action -> Results execution model
- Strict state separation (Configuration State vs. Executed Results State)
- Minimalistic academic research UI/UX with professional typography and restrained palette
- Comprehensive error handling and input validation
- Six dedicated research modules:
  1. Single-Agent UCB1 Deep Dive & Learning Convergence
  2. Multi-Algorithm Benchmark Arena (UCB1, Thompson Sampling, ε-Greedy, Random)
  3. Contextual Personalization (Disjoint LinUCB with User Persona Simulator)
  4. Non-Stationary Dynamics & Ad Fatigue Adaptation (D-UCB, SW-UCB)
  5. Unbiased Offline Policy Evaluation (Li et al. Replay Method)
  6. Theoretical Foundations & Academic Viva Defense
"""

import sys
import os

# Ensure project root is available on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.environment import AdvertisementEnvironment
from src.ucb_agent import UCB1Agent
from src.baseline_agents import RandomAgent, EpsilonGreedyAgent, ThompsonSamplingAgent
from src.contextual_environment import ContextualAdvertisementEnvironment
from src.contextual_agent import LinUCBAgent
from src.non_stationary import NonStationaryEnvironment, DiscountedUCBAgent, SlidingWindowUCBAgent
from src.offline_evaluator import generate_sample_ad_dataset, evaluate_policy_offline_replay
from src.simulation import (
    run_simulation,
    run_multi_agent_simulation,
    run_contextual_simulation,
    run_non_stationary_simulation,
)
from src.evaluation import calculate_metrics

# ------------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & RESTRAINED ACADEMIC DESIGN SYSTEM
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="AdUCB Lab — Dynamic Advertisement Selection",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        color: #0f172a;
    }
    
    .stApp {
        background-color: #f8fafc;
    }

    /* Constrain content to 1120px to eliminate excessive horizontal whitespace */
    .main .block-container {
        max-width: 1120px !important;
        padding-top: 1.0rem !important;
        padding-bottom: 3.5rem !important;
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
        margin: 0 auto !important;
    }

    /* Top Application Bar */
    .top-nav-bar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.5rem 0;
        border-bottom: 1px solid #e2e8f0;
        margin-bottom: 1.25rem;
    }
    .brand-group {
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }
    .brand-logo {
        font-size: 0.95rem;
        font-weight: 700;
        color: #0f172a;
        letter-spacing: -0.02em;
    }
    .brand-divider {
        color: #cbd5e1;
        font-size: 0.9rem;
    }
    .brand-sub {
        font-size: 0.82rem;
        font-weight: 500;
        color: #64748b;
    }

    /* Header Container */
    .header-container {
        margin-bottom: 1.5rem;
    }
    .project-title {
        font-size: 1.75rem;
        font-weight: 700;
        color: #0f172a;
        letter-spacing: -0.025em;
        margin: 0;
        line-height: 1.2;
    }
    .project-subtitle {
        font-size: 0.92rem;
        color: #475569;
        margin-top: 0.35rem;
        line-height: 1.5;
        max-width: 960px;
    }

    /* Status Badges */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        font-size: 0.78rem;
        font-weight: 600;
        padding: 0.2rem 0.65rem;
        border-radius: 9999px;
        letter-spacing: 0.01em;
    }
    .status-ready {
        background-color: #f1f5f9;
        color: #475569;
        border: 1px solid #cbd5e1;
    }
    .status-completed {
        background-color: #ecfdf5;
        color: #047857;
        border: 1px solid #a7f3d0;
    }
    .status-changed {
        background-color: #fffbeb;
        color: #b45309;
        border: 1px solid #fde68a;
    }

    /* Streamlit Tab Bar Customization (Minimalist Scientific Style) */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        border-bottom: 1px solid #e2e8f0 !important;
        padding-bottom: 0px;
        background-color: transparent;
        margin-bottom: 1.25rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 38px;
        padding: 0 14px;
        font-size: 13.5px;
        font-weight: 500;
        color: #64748b !important;
        border-radius: 4px 4px 0 0;
        border: none;
        background-color: transparent !important;
        transition: all 0.15s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #0f172a !important;
        background-color: #f1f5f9 !important;
    }
    .stTabs [aria-selected="true"] {
        color: #0f172a !important;
        font-weight: 600 !important;
    }
    div[data-baseweb="tab-highlight"] {
        background-color: #0f172a !important;
        height: 2px !important;
    }
    div[data-baseweb="tab-border"] {
        background-color: #e2e8f0 !important;
    }

    /* Container Borders */
    [data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 8px !important;
        border: 1px solid #e2e8f0 !important;
        background-color: #ffffff !important;
        padding: 1.25rem 1.5rem !important;
        margin-bottom: 1.25rem !important;
    }

    .section-label {
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #475569;
        margin-bottom: 0.6rem;
    }

    .sub-section-label {
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #64748b;
        margin-bottom: 0.35rem;
    }

    /* KPI Metric Blocks (Understated Bloomberg / Linear Style) */
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 0.85rem 1.05rem;
        text-align: left;
    }
    .metric-lbl {
        font-size: 0.70rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-val {
        font-size: 1.55rem;
        font-weight: 700;
        color: #0f172a;
        margin-top: 0.15rem;
        line-height: 1.2;
        font-variant-numeric: tabular-nums;
    }
    .metric-sub {
        font-size: 0.73rem;
        color: #475569;
        margin-top: 0.2rem;
    }

    /* Notice Banners */
    .config-changed-banner {
        background-color: #fffbeb;
        border: 1px solid #fde68a;
        border-left: 4px solid #d97706;
        padding: 0.75rem 1rem;
        border-radius: 6px;
        margin-bottom: 1.0rem;
        font-size: 0.85rem;
        color: #92400e;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Empty State Workbench Card */
    .empty-state-box {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 2.25rem 1.5rem;
        text-align: center;
        color: #64748b;
        margin: 1.0rem 0;
    }
    .empty-state-title {
        font-size: 1.05rem;
        font-weight: 600;
        color: #0f172a;
        margin-bottom: 0.35rem;
    }
    .empty-state-desc {
        font-size: 0.875rem;
        color: #475569;
        max-width: 540px;
        margin: 0 auto;
        line-height: 1.5;
    }

    /* Educational Theory Box */
    .theory-box {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 3px solid #334155;
        padding: 1.0rem 1.25rem;
        border-radius: 6px;
        margin-top: 1.5rem;
    }

    /* Buttons */
    button[kind="primary"] {
        background-color: #0f172a !important;
        color: #ffffff !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        border: 1px solid #0f172a !important;
        padding: 0.45rem 1.1rem !important;
        font-size: 0.875rem !important;
        transition: background-color 0.15s ease;
    }
    button[kind="primary"]:hover {
        background-color: #1e293b !important;
        border-color: #1e293b !important;
    }

    button[kind="secondary"] {
        background-color: #ffffff !important;
        color: #475569 !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 6px !important;
        font-weight: 500 !important;
        padding: 0.45rem 1.0rem !important;
        font-size: 0.875rem !important;
        transition: all 0.15s ease;
    }
    button[kind="secondary"]:hover {
        background-color: #f8fafc !important;
        color: #0f172a !important;
        border-color: #94a3b8 !important;
    }

    /* Clean Streamlit Inputs */
    div[data-baseweb="input"] {
        border-radius: 6px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------------------
# 2. SESSION STATE MANAGEMENT (STRICT CONFIGURATION VS. RESULTS SEPARATION)
# ------------------------------------------------------------------------------
if "ucb_results" not in st.session_state:
    st.session_state.ucb_results = None
if "ucb_executed_config" not in st.session_state:
    st.session_state.ucb_executed_config = None
if "ucb_status" not in st.session_state:
    st.session_state.ucb_status = "Ready"

if "arena_results" not in st.session_state:
    st.session_state.arena_results = None
if "arena_executed_config" not in st.session_state:
    st.session_state.arena_executed_config = None

if "contextual_results" not in st.session_state:
    st.session_state.contextual_results = None
if "contextual_executed_config" not in st.session_state:
    st.session_state.contextual_executed_config = None

if "non_stationary_results" not in st.session_state:
    st.session_state.non_stationary_results = None
if "non_stationary_executed_config" not in st.session_state:
    st.session_state.non_stationary_executed_config = None

if "offline_results" not in st.session_state:
    st.session_state.offline_results = None

# Default parameter constants
DEFAULT_ROUNDS = 10000
DEFAULT_C = 2.0
DEFAULT_SEED = 42
DEFAULT_CTRS = [0.05, 0.10, 0.03, 0.15, 0.08]
AD_NAMES = ["Ad 1", "Ad 2", "Ad 3", "Ad 4", "Ad 5"]


def apply_academic_plot_style(ax, title="", xlabel="", ylabel=""):
    """Applies clean, publication-grade academic styling to a matplotlib axis."""
    ax.set_facecolor("#ffffff")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cbd5e1")
    ax.spines["bottom"].set_color("#cbd5e1")
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(colors="#475569", labelsize=8.5)
    if title:
        ax.set_title(title, fontweight="bold", fontsize=10.0, pad=8, color="#0f172a", loc="left")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=8.8, color="#475569", labelpad=5)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8.8, color="#475569", labelpad=5)
    ax.grid(True, linestyle="--", linewidth=0.6, color="#f1f5f9", alpha=0.9)


# ------------------------------------------------------------------------------
# 3. TOP NAVIGATION APP BAR & COMPACT HEADER
# ------------------------------------------------------------------------------
status_text = st.session_state.ucb_status
if "completed" in status_text.lower():
    badge_html = f'<div class="status-badge status-completed">● {status_text}</div>'
elif "changed" in status_text.lower():
    badge_html = f'<div class="status-badge status-changed">● {status_text}</div>'
else:
    badge_html = f'<div class="status-badge status-ready">● {status_text}</div>'

st.markdown(
    f"""
    <div class="top-nav-bar">
        <div class="brand-group">
            <span class="brand-logo">AdUCB Lab</span>
            <span class="brand-divider">/</span>
            <span class="brand-sub">Reinforcement Learning Research Environment</span>
        </div>
        <div>
            {badge_html}
        </div>
    </div>
    <div class="header-container">
        <h1 class="project-title">Dynamic Advertisement Selection</h1>
        <div class="project-subtitle">
            Empirical experimentation platform for evaluating Multi-Armed Bandit exploration–exploitation algorithms,
            sublinear regret bounds, and personalized contextual ad allocation.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------------------
# 4. NAVIGATION TABS
# ------------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Single-Agent UCB1",
    "Algorithm Arena",
    "Contextual Personalization",
    "Non-Stationary Dynamics",
    "Offline Evaluation",
    "Methodology & Viva Guide",
])

# ==============================================================================
# TAB 1: SINGLE-AGENT UCB1 SIMULATION (EXPLICIT USER-CONTROLLED EXECUTION)
# ==============================================================================
with tab1:
    st.markdown('<div class="section-label">1. Experiment Configuration Panel</div>', unsafe_allow_html=True)
    
    with st.container(border=True):
        st.markdown('<div class="sub-section-label">Simulation Hyperparameters</div>', unsafe_allow_html=True)
        cfg_col1, cfg_col2, cfg_col3 = st.columns([1.2, 1.2, 1.0])
        with cfg_col1:
            input_rounds = st.number_input(
                "Number of Rounds (T)",
                min_value=100,
                max_value=100000,
                value=DEFAULT_ROUNDS,
                step=1000,
                help="Total sequential ad impressions evaluated in the closed-loop simulation.",
                key="input_ucb_rounds"
            )
        with cfg_col2:
            input_c = st.number_input(
                "Exploration Coefficient (c)",
                min_value=0.01,
                max_value=10.0,
                value=DEFAULT_C,
                step=0.1,
                format="%.2f",
                help="Controls the width of the confidence bound derived from Hoeffding's Inequality.",
                key="input_ucb_c"
            )
        with cfg_col3:
            input_seed = st.number_input(
                "Random Seed",
                min_value=0,
                max_value=999999,
                value=DEFAULT_SEED,
                step=1,
                help="Ensures reproducible stochastic click sampling.",
                key="input_ucb_seed"
            )

        st.markdown('<div style="height: 6px;"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-section-label">Simulated Environment Ground Truth (Hidden from Agent)</div>', unsafe_allow_html=True)
        st.caption("True Click-Through Rates that govern stochastic click generation. The agent must discover these solely through feedback.")
        col_ad1, col_ad2, col_ad3, col_ad4, col_ad5 = st.columns(5)
        c1 = col_ad1.number_input("Ad 1 CTR", min_value=0.00, max_value=1.00, value=DEFAULT_CTRS[0], step=0.01, format="%.2f", key="ctr_ad1")
        c2 = col_ad2.number_input("Ad 2 CTR", min_value=0.00, max_value=1.00, value=DEFAULT_CTRS[1], step=0.01, format="%.2f", key="ctr_ad2")
        c3 = col_ad3.number_input("Ad 3 CTR", min_value=0.00, max_value=1.00, value=DEFAULT_CTRS[2], step=0.01, format="%.2f", key="ctr_ad3")
        c4 = col_ad4.number_input("Ad 4 CTR (Optimal)", min_value=0.00, max_value=1.00, value=DEFAULT_CTRS[3], step=0.01, format="%.2f", key="ctr_ad4")
        c5 = col_ad5.number_input("Ad 5 CTR", min_value=0.00, max_value=1.00, value=DEFAULT_CTRS[4], step=0.01, format="%.2f", key="ctr_ad5")
        
        current_ctrs = [c1, c2, c3, c4, c5]
        
        # Current active configuration snapshot
        current_config = {
            "rounds": int(input_rounds),
            "c": float(input_c),
            "seed": int(input_seed),
            "ctrs": current_ctrs,
        }

        st.markdown('<div style="height: 10px;"></div>', unsafe_allow_html=True)
        btn_col1, btn_col2, _ = st.columns([1.6, 1.2, 5.0])
        with btn_col1:
            run_clicked = st.button("▶ Run Simulation", type="primary", use_container_width=True)
        with btn_col2:
            reset_clicked = st.button("↺ Reset", type="secondary", use_container_width=True)

    # Handle Reset Action
    if reset_clicked:
        st.session_state.ucb_results = None
        st.session_state.ucb_executed_config = None
        st.session_state.ucb_status = "Ready"

    # Handle Run Simulation Action
    if run_clicked:
        # 1. Input Validation
        validation_errors = []
        if current_config["rounds"] < 100:
            validation_errors.append("Number of rounds must be at least 100.")
        if current_config["c"] <= 0:
            validation_errors.append("Exploration coefficient (c) must be strictly greater than 0.")
        for idx, ctr in enumerate(current_ctrs):
            if not (0.0 <= ctr <= 1.0):
                validation_errors.append(f"CTR for Ad {idx+1} must be between 0.0 and 1.0.")

        if validation_errors:
            st.session_state.ucb_status = "Error"
            for err in validation_errors:
                st.error(f"Validation Error: {err}")
        else:
            with st.spinner("Executing closed-loop UCB1 simulation..."):
                env = AdvertisementEnvironment(
                    true_ctrs=current_config["ctrs"],
                    ad_names=AD_NAMES,
                    seed=current_config["seed"],
                )
                agent = UCB1Agent(num_ads=len(current_config["ctrs"]), c=current_config["c"])
                df_sim = run_simulation(
                    env=env,
                    agent=agent,
                    num_rounds=current_config["rounds"],
                    seed=current_config["seed"],
                )
                metrics = calculate_metrics(df_sim, env, agent)

                # Persist in session state
                st.session_state.ucb_results = {
                    "df": df_sim,
                    "metrics": metrics,
                    "env": env,
                    "agent": agent,
                }
                st.session_state.ucb_executed_config = dict(current_config)
                st.session_state.ucb_status = f"Simulation completed (T={current_config['rounds']:,})"

    # --------------------------------------------------------------------------
    # RESULTS RENDERING & CONFIGURATION CHANGE DETECTION
    # --------------------------------------------------------------------------
    st.markdown('<div class="section-label" style="margin-top: 1.25rem;">2. Simulation Analytics & Results</div>', unsafe_allow_html=True)

    if st.session_state.ucb_results is None:
        # Required Initial State: No results or fake data before execution
        st.markdown(
            """
            <div class="empty-state-box">
                <div class="empty-state-title">No simulation results yet</div>
                <div class="empty-state-desc">
                    Configure the parameters above and click <strong>Run Simulation</strong> 
                    to execute the UCB1 multi-armed bandit algorithm and generate analytical results.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        executed_config = st.session_state.ucb_executed_config
        results = st.session_state.ucb_results
        metrics = results["metrics"]
        df_sim = results["df"]

        # Check if user has modified configuration after simulation was run
        is_config_modified = (current_config != executed_config)
        if is_config_modified:
            st.markdown(
                f"""
                <div class="config-changed-banner">
                    <span>⚠️</span>
                    <span>
                        <strong>Configuration changed:</strong> Input parameters modified (T={current_config['rounds']:,}, c={current_config['c']}). 
                        Showing results for last executed run (T={executed_config['rounds']:,}, c={executed_config['c']}). 
                        Click <strong>Run Simulation</strong> to update results.
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # KPI Metrics Row
        k_col1, k_col2, k_col3, k_col4 = st.columns(4)
        with k_col1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-lbl">Total Reward (Clicks)</div>
                    <div class="metric-val">{metrics['total_reward']:,}</div>
                    <div class="metric-sub">Across {executed_config['rounds']:,} rounds</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with k_col2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-lbl">Average System CTR</div>
                    <div class="metric-val">{metrics['average_reward']*100:.2f}%</div>
                    <div class="metric-sub">Empirical click yield</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with k_col3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-lbl">Optimal Advertisement</div>
                    <div class="metric-val">{metrics['best_ad_name']}</div>
                    <div class="metric-sub">True CTR: {metrics['max_true_ctr']*100:.1f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with k_col4:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-lbl">Cumulative Regret</div>
                    <div class="metric-val">{metrics['total_expected_regret']:.2f}</div>
                    <div class="metric-sub">Sublinear deficit growth</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Charts Section
        st.markdown('<div style="height: 14px;"></div>', unsafe_allow_html=True)
        plot_col1, plot_col2 = st.columns(2)

        with plot_col1:
            fig1, ax1 = plt.subplots(figsize=(5.6, 3.3), facecolor="#ffffff", dpi=140)
            apply_academic_plot_style(ax1, "Cumulative Reward Over Time", "Simulation Round (t)", "Cumulative Clicks")
            ax1.plot(df_sim["round"], df_sim["cumulative_reward"], color="#1e40af", linewidth=1.8, label="Harvested Clicks")
            ax1.legend(loc="upper left", frameon=True, facecolor="#ffffff", edgecolor="#e2e8f0", fontsize=8.5)
            plt.tight_layout()
            st.pyplot(fig1, width="stretch")
            plt.close(fig1)

        with plot_col2:
            fig2, ax2 = plt.subplots(figsize=(5.6, 3.3), facecolor="#ffffff", dpi=140)
            apply_academic_plot_style(ax2, "Cumulative Regret (Sublinear O(ln T))", "Simulation Round (t)", "Expected Regret")
            ax2.plot(df_sim["round"], df_sim["cumulative_expected_regret"], color="#b91c1c", linewidth=1.8, label="Cumulative Regret")
            ax2.legend(loc="upper left", frameon=True, facecolor="#ffffff", edgecolor="#e2e8f0", fontsize=8.5)
            plt.tight_layout()
            st.pyplot(fig2, width="stretch")
            plt.close(fig2)

        plot_col3, plot_col4 = st.columns(2)
        
        with plot_col3:
            fig3, ax3 = plt.subplots(figsize=(5.6, 3.3), facecolor="#ffffff", dpi=140)
            apply_academic_plot_style(ax3, "Advertisement Selection Frequency", "Ad Creative", "Total Impressions")
            best_idx = int(np.argmax(executed_config["ctrs"]))
            colors = ["#047857" if i == best_idx else "#475569" for i in range(len(AD_NAMES))]
            counts = metrics["selection_counts"]
            bars = ax3.bar(AD_NAMES, counts, color=colors, edgecolor="#0f172a", linewidth=0.6, alpha=0.9, width=0.48)
            for b in bars:
                h = b.get_height()
                ax3.annotate(f"{int(h):,}", xy=(b.get_x() + b.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8, fontweight="bold", color="#0f172a")
            plt.tight_layout()
            st.pyplot(fig3, width="stretch")
            plt.close(fig3)

        with plot_col4:
            fig4, ax4 = plt.subplots(figsize=(5.6, 3.3), facecolor="#ffffff", dpi=140)
            apply_academic_plot_style(ax4, "True CTR vs. Estimated CTR Convergence", "Ad Creative", "Click-Through Rate")
            x_idx = np.arange(len(AD_NAMES))
            w = 0.32
            ax4.bar(x_idx - w/2, executed_config["ctrs"], w, label="True CTR (Ground Truth)", color="#94a3b8", alpha=0.9, edgecolor="#0f172a", linewidth=0.5)
            ax4.bar(x_idx + w/2, metrics["estimated_ctrs"], w, label="Estimated CTR (UCB1)", color="#d97706", alpha=0.9, edgecolor="#0f172a", linewidth=0.5)
            ax4.set_xticks(x_idx)
            ax4.set_xticklabels(AD_NAMES, color="#0f172a")
            ax4.legend(loc="upper right", frameon=True, facecolor="#ffffff", edgecolor="#e2e8f0", fontsize=8.5)
            plt.tight_layout()
            st.pyplot(fig4, width="stretch")
            plt.close(fig4)

        # Performance Data Table
        st.markdown('<div class="section-label" style="margin-top: 1.25rem;">3. Advertisement Performance Matrix</div>', unsafe_allow_html=True)
        summary_table = metrics["ad_summary_df"].copy()
        summary_table["True CTR"] = summary_table["True CTR"].apply(lambda v: f"{v*100:.2f}%")
        summary_table["Estimated CTR"] = summary_table["Estimated CTR"].apply(lambda v: f"{v*100:.2f}%")
        summary_table["Absolute Error"] = summary_table["Absolute Error"].apply(lambda v: f"{v*100:.3f}%")
        summary_table["Selection %"] = summary_table["Selection %"].apply(lambda v: f"{v:.2f}%")
        summary_table["Selections"] = summary_table["Selections"].apply(lambda v: f"{v:,}")
        summary_table["Total Clicks"] = summary_table["Total Clicks"].apply(lambda v: f"{v:,}")
        st.dataframe(summary_table, width="stretch", hide_index=True)

    # Educational Theory Section
    st.markdown(
        """
        <div class="theory-box">
            <div style="font-weight: 700; color: #0f172a; margin-bottom: 0.35rem; font-size: 0.92rem;">Theoretical Decision Rule: UCB1 (Auer et al., 2002)</div>
            <div style="font-size: 0.85rem; color: #334155; line-height: 1.55;">
                The agent maximizes the upper confidence bound at every decision round:
                <br>
                <code>UCB<sub>i</sub>(t) = μ̂<sub>i</sub>(t-1) + c · √( 2 · ln(t) / N<sub>i</sub>(t-1) )</code>
                <br><br>
                • <strong>Exploitation Term (μ̂<sub>i</sub>):</strong> Prioritizes ad creatives with historically verified high empirical click yields.<br>
                • <strong>Exploration Uncertainty Bonus (√(ln(t)/N<sub>i</sub>)):</strong> Expands for rarely visited arms, guaranteeing that under-explored creatives are systematically tested.<br>
                • <strong>Regret Bound:</strong> Proven to attain asymptotic logarithmic regret <em>O(ln T)</em>, achieving the Lai-Robbins theoretical lower bound.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# TAB 2: MULTI-ALGORITHM BENCHMARK ARENA (EXPLICIT USER CONTROL)
# ==============================================================================
with tab2:
    st.markdown('<div class="section-label">1. Arena Benchmark Configuration</div>', unsafe_allow_html=True)
    
    with st.container(border=True):
        a_col1, a_col2 = st.columns(2)
        with a_col1:
            arena_rounds = st.number_input(
                "Benchmark Horizon (Rounds)",
                min_value=500,
                max_value=50000,
                value=10000,
                step=1000,
                key="input_arena_rounds"
            )
        with a_col2:
            arena_seed = st.number_input(
                "Benchmark Random Seed",
                min_value=0,
                max_value=999999,
                value=42,
                step=1,
                key="input_arena_seed"
            )

        st.markdown('<div style="height: 6px;"></div>', unsafe_allow_html=True)
        btn_a1, btn_a2, _ = st.columns([1.8, 1.2, 5.0])
        with btn_a1:
            run_arena_clicked = st.button("⚔️ Run Benchmark Arena", type="primary", use_container_width=True)
        with btn_a2:
            reset_arena_clicked = st.button("↺ Reset Arena", type="secondary", use_container_width=True)

    if reset_arena_clicked:
        st.session_state.arena_results = None
        st.session_state.arena_executed_config = None

    if run_arena_clicked:
        with st.spinner(f"Benchmarking 5 policies over {arena_rounds:,} rounds..."):
            agents = {
                "UCB1 (c=2.0)": UCB1Agent(num_ads=5, c=2.0),
                "Thompson Sampling": ThompsonSamplingAgent(num_ads=5, seed=int(arena_seed)),
                "ε-Greedy (decay)": EpsilonGreedyAgent(num_ads=5, epsilon=0.5, decay=True, seed=int(arena_seed)),
                "ε-Greedy (fixed ε=0.1)": EpsilonGreedyAgent(num_ads=5, epsilon=0.1, decay=False, seed=int(arena_seed)),
                "Static Random A/B": RandomAgent(num_ads=5, seed=int(arena_seed)),
            }
            results_arena = run_multi_agent_simulation(
                agents=agents,
                true_ctrs=DEFAULT_CTRS,
                num_rounds=int(arena_rounds),
                seed=int(arena_seed)
            )
            st.session_state.arena_results = results_arena
            st.session_state.arena_executed_config = {"rounds": int(arena_rounds), "seed": int(arena_seed)}

    st.markdown('<div class="section-label" style="margin-top: 1.25rem;">2. Head-to-Head Comparative Leaderboard</div>', unsafe_allow_html=True)
    
    if st.session_state.arena_results is None:
        st.markdown(
            """
            <div class="empty-state-box">
                <div class="empty-state-title">No benchmark results yet</div>
                <div class="empty-state-desc">
                    Configure the simulation horizon and click <strong>Run Benchmark Arena</strong> 
                    to compare UCB1, Thompson Sampling, ε-Greedy, and Static A/B Testing side-by-side.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        results_arena = st.session_state.arena_results
        exec_cfg = st.session_state.arena_executed_config

        if int(arena_rounds) != exec_cfg["rounds"] or int(arena_seed) != exec_cfg["seed"]:
            st.markdown(
                f"""
                <div class="config-changed-banner">
                    <span>⚠️</span>
                    <span>Configuration changed. Displaying results for T={exec_cfg['rounds']:,}. Click <strong>Run Benchmark Arena</strong> to update.</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Compute Leaderboard Metrics
        lb_rows = []
        for name, df_a in results_arena.items():
            total_c = int(df_a["cumulative_reward"].iloc[-1])
            regret_val = float(df_a["cumulative_expected_regret"].iloc[-1]) if "cumulative_expected_regret" in df_a else float(df_a["cumulative_regret"].iloc[-1])
            lb_rows.append({
                "Algorithm Policy": name,
                "Total Clicks": total_c,
                "Effective System CTR": f"{(total_c / len(df_a)) * 100:.2f}%",
                "Cumulative Regret": round(regret_val, 2),
                "Regret Profile": "Sublinear O(ln T)" if ("UCB" in name or "Thompson" in name or "decay" in name) else "Linear O(T)",
            })

        df_lb = pd.DataFrame(lb_rows).sort_values(by="Total Clicks", ascending=False)
        st.dataframe(df_lb, width="stretch", hide_index=True)

        ar_col1, ar_col2 = st.columns(2)
        policy_palette = {
            "UCB1 (c=2.0)": "#1e40af",
            "Thompson Sampling": "#047857",
            "ε-Greedy (decay)": "#d97706",
            "ε-Greedy (fixed ε=0.1)": "#9333ea",
            "Static Random A/B": "#94a3b8",
        }

        with ar_col1:
            fig_ar1, ax_ar1 = plt.subplots(figsize=(5.6, 3.3), facecolor="#ffffff", dpi=140)
            apply_academic_plot_style(ax_ar1, "Cumulative Regret Comparison", "Simulation Round (t)", "Cumulative Regret")
            for name, df_a in results_arena.items():
                reg_c = "cumulative_expected_regret" if "cumulative_expected_regret" in df_a else "cumulative_regret"
                ax_ar1.plot(df_a["round"], df_a[reg_c], label=name, color=policy_palette.get(name, "#475569"), linewidth=1.8)
            ax_ar1.legend(fontsize=8, frameon=True, facecolor="#ffffff", edgecolor="#e2e8f0")
            plt.tight_layout()
            st.pyplot(fig_ar1, width="stretch")
            plt.close(fig_ar1)

        with ar_col2:
            fig_ar2, ax_ar2 = plt.subplots(figsize=(5.6, 3.3), facecolor="#ffffff", dpi=140)
            apply_academic_plot_style(ax_ar2, "Harvested Clicks Over Time", "Simulation Round (t)", "Total Clicks")
            for name, df_a in results_arena.items():
                ax_ar2.plot(df_a["round"], df_a["cumulative_reward"], label=name, color=policy_palette.get(name, "#475569"), linewidth=1.8)
            ax_ar2.legend(fontsize=8, frameon=True, facecolor="#ffffff", edgecolor="#e2e8f0")
            plt.tight_layout()
            st.pyplot(fig_ar2, width="stretch")
            plt.close(fig_ar2)


# ==============================================================================
# TAB 3: CONTEXTUAL PERSONALIZATION (LINUCB) (EXPLICIT USER CONTROL)
# ==============================================================================
with tab3:
    st.markdown('<div class="section-label">1. Contextual Personalization Configuration</div>', unsafe_allow_html=True)
    
    with st.container(border=True):
        ctx_col1, ctx_col2 = st.columns([1.2, 1.2])
        with ctx_col1:
            ctx_rounds = st.number_input("Contextual Rounds", min_value=1000, max_value=50000, value=10000, step=1000, key="input_ctx_rounds")
        with ctx_col2:
            ctx_alpha = st.number_input("LinUCB Confidence Parameter (α)", min_value=0.1, max_value=5.0, value=1.0, step=0.1, key="input_ctx_alpha")

        st.markdown('<div style="height: 6px;"></div>', unsafe_allow_html=True)
        btn_cx1, btn_cx2, _ = st.columns([2.0, 1.2, 4.8])
        with btn_cx1:
            run_ctx_clicked = st.button("🎯 Run LinUCB Personalization", type="primary", use_container_width=True)
        with btn_cx2:
            reset_ctx_clicked = st.button("↺ Reset LinUCB", type="secondary", use_container_width=True)

    if reset_ctx_clicked:
        st.session_state.contextual_results = None
        st.session_state.contextual_executed_config = None

    if run_ctx_clicked:
        with st.spinner("Simulating personalized visitor streams with LinUCB..."):
            c_env = ContextualAdvertisementEnvironment(num_ads=4, context_dim=5, seed=42)
            c_agent = LinUCBAgent(num_ads=4, context_dim=5, alpha=float(ctx_alpha))
            df_ctx = run_contextual_simulation(c_env, c_agent, num_rounds=int(ctx_rounds), seed=42)
            st.session_state.contextual_results = {
                "df": df_ctx,
                "thetas": c_agent.get_estimated_thetas(),
                "env": c_env,
                "agent": c_agent,
            }
            st.session_state.contextual_executed_config = {"rounds": int(ctx_rounds), "alpha": float(ctx_alpha)}

    st.markdown('<div class="section-label" style="margin-top: 1.25rem;">2. Interactive User Persona Simulator</div>', unsafe_allow_html=True)
    
    with st.container(border=True):
        st.markdown('<div class="sub-section-label">Visitor Attribute Selector (Context Vector x ∈ ℝ⁵)</div>', unsafe_allow_html=True)
        p_c1, p_c2, p_c3, p_c4 = st.columns(4)
        u_mobile = p_c1.checkbox("Mobile Device", value=True)
        u_young = p_c2.checkbox("Age Under 30", value=True)
        u_evening = p_c3.checkbox("Evening Browsing", value=False)
        u_tech = p_c4.checkbox("High Tech Affinity", value=True)

        context_vector = np.array([
            1.0,  # Bias
            1.0 if u_mobile else 0.0,
            1.0 if u_young else 0.0,
            1.0 if u_evening else 0.0,
            1.0 if u_tech else 0.0,
        ], dtype=np.float64)

        # Baseline environment for persona inspection
        demo_env = ContextualAdvertisementEnvironment(num_ads=4, context_dim=5, seed=42)
        pred_probs = demo_env.get_ad_click_probabilities(context_vector)
        rec_ad_idx = int(np.argmax(pred_probs))

        st.markdown(
            f"""
            <div style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; padding: 0.8rem 1.0rem; margin-top: 0.75rem;">
                <div style="font-size: 0.72rem; font-family: monospace; color: #64748b; margin-bottom: 0.25rem;">Context Vector x = [1 (bias), {int(context_vector[1])} (mobile), {int(context_vector[2])} (young), {int(context_vector[3])} (evening), {int(context_vector[4])} (tech)]^T</div>
                <span style="font-size: 0.88rem; font-weight: 600; color: #334155;">Recommended Creative:</span>
                <span style="font-size: 1.05rem; font-weight: 700; color: #0284c7; margin-left: 0.4rem;">{demo_env.ad_names[rec_ad_idx]}</span>
                <span style="font-size: 0.85rem; color: #047857; font-weight: 600; margin-left: 0.5rem;">(Expected CTR: {pred_probs[rec_ad_idx]*100:.1f}%)</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-label" style="margin-top: 1.25rem;">3. Personalization Results</div>', unsafe_allow_html=True)
    
    if st.session_state.contextual_results is None:
        st.markdown(
            """
            <div class="empty-state-box">
                <div class="empty-state-title">No contextual simulation results yet</div>
                <div class="empty-state-desc">
                    Click <strong>Run LinUCB Personalization</strong> above to simulate multi-dimensional 
                    visitor feature vectors and evaluate personalized advertisement targeting.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        df_ctx = st.session_state.contextual_results["df"]
        total_clicks_ctx = int(df_ctx["cumulative_reward"].iloc[-1])
        system_ctr_ctx = (total_clicks_ctx / len(df_ctx)) * 100

        k_cx1, k_cx2, k_cx3 = st.columns(3)
        with k_cx1:
            st.markdown(f'<div class="metric-card"><div class="metric-lbl">Personalized Clicks</div><div class="metric-val">{total_clicks_ctx:,}</div><div class="metric-sub">Closed-loop conversions</div></div>', unsafe_allow_html=True)
        with k_cx2:
            st.markdown(f'<div class="metric-card"><div class="metric-lbl">Personalized System CTR</div><div class="metric-val">{system_ctr_ctx:.2f}%</div><div class="metric-sub">Feature-conditioned yield</div></div>', unsafe_allow_html=True)
        with k_cx3:
            st.markdown(f'<div class="metric-card"><div class="metric-lbl">Contextual Regret</div><div class="metric-val">{df_ctx["cumulative_regret"].iloc[-1]:.2f}</div><div class="metric-sub">Cumulative policy deficit</div></div>', unsafe_allow_html=True)


# ==============================================================================
# TAB 4: NON-STATIONARY DYNAMICS & AD FATIGUE (EXPLICIT USER CONTROL)
# ==============================================================================
with tab4:
    st.markdown('<div class="section-label">1. Non-Stationary Shift Configuration</div>', unsafe_allow_html=True)
    
    with st.container(border=True):
        ns_col1, ns_col2 = st.columns(2)
        with ns_col1:
            shift_round = st.number_input("Fatigue Change Point (Round)", min_value=1000, max_value=9000, value=5000, step=500, key="input_shift_point")
        with ns_col2:
            discount_gamma = st.number_input("D-UCB Discount Factor (γ)", min_value=0.900, max_value=0.9999, value=0.998, step=0.001, format="%.3f", key="input_gamma")

        st.caption("At round τ, Ad 4 collapses from 15% to 2% (ad fatigue), while Ad 2 surges from 10% to 18%.")

        st.markdown('<div style="height: 6px;"></div>', unsafe_allow_html=True)
        btn_ns1, btn_ns2, _ = st.columns([2.0, 1.2, 4.8])
        with btn_ns1:
            run_ns_clicked = st.button("🔄 Run Non-Stationary Simulation", type="primary", use_container_width=True)
        with btn_ns2:
            reset_ns_clicked = st.button("↺ Reset Non-Stationary", type="secondary", use_container_width=True)

    if reset_ns_clicked:
        st.session_state.non_stationary_results = None
        st.session_state.non_stationary_executed_config = None

    if run_ns_clicked:
        with st.spinner("Simulating non-stationary regime shift..."):
            env_ns = NonStationaryEnvironment(
                initial_ctrs=DEFAULT_CTRS,
                change_points={int(shift_round): [0.05, 0.18, 0.03, 0.02, 0.08]},
                seed=42,
            )
            agents_ns = {
                "Stationary UCB1": UCB1Agent(num_ads=5, c=1.0),
                "Discounted-UCB (D-UCB)": DiscountedUCBAgent(num_ads=5, gamma=float(discount_gamma), c=0.2),
                "Sliding-Window UCB (SW-UCB)": SlidingWindowUCBAgent(num_ads=5, window_size=1500, c=0.2),
            }
            res_ns = run_non_stationary_simulation(env_ns, agents_ns, num_rounds=10000, seed=42)
            st.session_state.non_stationary_results = res_ns
            st.session_state.non_stationary_executed_config = {"shift": int(shift_round), "gamma": float(discount_gamma)}

    st.markdown('<div class="section-label" style="margin-top: 1.25rem;">2. Non-Stationary Shift Analytics</div>', unsafe_allow_html=True)
    
    if st.session_state.non_stationary_results is None:
        st.markdown(
            """
            <div class="empty-state-box">
                <div class="empty-state-title">No non-stationary simulation results yet</div>
                <div class="empty-state-desc">
                    Click <strong>Run Non-Stationary Simulation</strong> above to observe how 
                    Discounted-UCB and Sliding-Window UCB adapt when ad performance degrades over time.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        res_ns = st.session_state.non_stationary_results
        exec_ns = st.session_state.non_stationary_executed_config

        fig_ns, ax_ns = plt.subplots(figsize=(8.8, 3.6), facecolor="#ffffff", dpi=140)
        apply_academic_plot_style(ax_ns, "Regret Accumulation Under Ad Fatigue Shift", "Simulation Round (t)", "Cumulative Regret")
        colors_ns = {"Stationary UCB1": "#b91c1c", "Discounted-UCB (D-UCB)": "#1e40af", "Sliding-Window UCB (SW-UCB)": "#047857"}
        for name, df_ns in res_ns.items():
            ax_ns.plot(df_ns["round"], df_ns["cumulative_regret"], label=name, color=colors_ns.get(name, "#475569"), linewidth=1.8)
        
        ax_ns.axvline(x=exec_ns["shift"], color="#0f172a", linestyle=":", linewidth=1.5, label=f"Fatigue Shift (t={exec_ns['shift']:,})")
        ax_ns.legend(loc="upper left", frameon=True, facecolor="#ffffff", edgecolor="#e2e8f0", fontsize=8.5)
        plt.tight_layout()
        st.pyplot(fig_ns, width="stretch")
        plt.close(fig_ns)


# ==============================================================================
# TAB 5: UNBIASED OFFLINE POLICY EVALUATION (EXPLICIT USER CONTROL)
# ==============================================================================
with tab5:
    st.markdown('<div class="section-label">1. Offline Replay Methodology (Li et al., WWW 2010)</div>', unsafe_allow_html=True)
    
    st.markdown(
        """
        <div style="font-size: 0.88rem; color: #475569; margin-bottom: 1.0rem; line-height: 1.5;">
            In production advertising, deploying untested RL agents on live traffic risks substantial revenue losses. 
            The <strong>Replay Method</strong> provides statistically valid, unbiased counterfactual evaluation 
            on historical impression logs without requiring live intervention.
        </div>
        """,
        unsafe_allow_html=True,
    )

    dataset_path = "data/real_ad_clicks.csv"
    if not os.path.exists(dataset_path):
        generate_sample_ad_dataset(output_path=dataset_path, num_samples=10000, seed=42)

    df_logs = pd.read_csv(dataset_path)

    with st.container(border=True):
        st.markdown('<div class="sub-section-label">Logged Impressions Dataset</div>', unsafe_allow_html=True)
        st.caption("10,000 historical ad impression events logged with uniform random exploration policy.")
        btn_off1, _ = st.columns([2.4, 5.6])
        with btn_off1:
            run_offline_clicked = st.button("📊 Run Offline Replay Evaluation", type="primary", use_container_width=True)

    if run_offline_clicked:
        with st.spinner("Replaying historical impression logs..."):
            u_agent = UCB1Agent(num_ads=4, c=1.5)
            t_agent = ThompsonSamplingAgent(num_ads=4, seed=42)
            l_agent = LinUCBAgent(num_ads=4, context_dim=5, alpha=1.0)

            ev_u = evaluate_policy_offline_replay(u_agent, df_logs, is_contextual=False)
            ev_t = evaluate_policy_offline_replay(t_agent, df_logs, is_contextual=False)
            ev_l = evaluate_policy_offline_replay(l_agent, df_logs, is_contextual=True)

            st.session_state.offline_results = {
                "UCB1": ev_u,
                "Thompson Sampling": ev_t,
                "Contextual LinUCB": ev_l,
            }

    st.markdown('<div class="section-label" style="margin-top: 1.25rem;">2. Offline Evaluation Summary</div>', unsafe_allow_html=True)
    
    if st.session_state.offline_results is None:
        st.markdown(
            """
            <div class="empty-state-box">
                <div class="empty-state-title">No offline evaluation results yet</div>
                <div class="empty-state-desc">
                    Click <strong>Run Offline Replay Evaluation</strong> above to evaluate UCB1, Thompson Sampling, 
                    and LinUCB against 10,000 historical logged impression records.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        off_res = st.session_state.offline_results
        off_summary_table = pd.DataFrame([
            {"Evaluated Policy": k, "Accepted Impressions": f"{v['accepted_rounds']:,}", "Clicks Harvested": v['total_clicks'], "Offline Replay CTR": f"{v['effective_ctr']*100:.2f}%"}
            for k, v in off_res.items()
        ])
        st.dataframe(off_summary_table, width="stretch", hide_index=True)


# ==============================================================================
# TAB 6: METHODOLOGY & ACADEMIC VIVA DEFENSE
# ==============================================================================
with tab6:
    st.markdown('<div class="section-label">Academic Viva Defense Handbook</div>', unsafe_allow_html=True)
    
    viva_questions = [
        (
            "What is the mathematical formulation of UCB1 and how does it balance exploration vs. exploitation?",
            "UCB1 evaluates: UCB_i(t) = μ̂_i + c · √(ln(t) / N_i). The empirical mean μ̂_i represents exploitation. The bonus term c · √(ln(t)/N_i) represents statistical uncertainty derived from Hoeffding's Inequality. As an ad is repeatedly chosen, N_i grows linearly, shrinking the exploration radius; if an ad is ignored, ln(t) slowly inflates the radius, ensuring it gets revisited."
        ),
        (
            "Why is cumulative regret logarithmic O(ln T) rather than linear O(T)?",
            "A static or non-learning policy pulls sub-optimal arms at a constant rate, yielding linear regret O(T). The Lai-Robbins theorem (1985) establishes that consistent bandit policies can achieve at best O(ln T) regret. UCB1 achieves this theoretical lower bound because sub-optimal arm allocation grows strictly with ln(t) / Δ_i²."
        ),
        (
            "How does LinUCB incorporate user context for personalized recommendation?",
            "Standard MAB assumes all users share identical click distributions. In Contextual Bandits (Li et al., WWW 2010), each user arrives with a feature vector x_t in R^d (e.g., mobile device, age group, browsing hour). LinUCB fits an online ridge regression model per ad creative, choosing the action with highest upper confidence score: p_(t,a) = x_t^T θ̂_a + α · √(x_t^T A_a^(-1) x_t)."
        ),
        (
            "How does Discounted-UCB adapt to non-stationary ad fatigue?",
            "Standard UCB gives equal weight 1 to all historical rounds, causing it to lag when an optimal ad fatiguing shift occurs. Discounted-UCB (Garivier & Moulines, 2011) applies a geometric discount factor γ in (0, 1) (e.g., γ = 0.998) to historical counts and rewards, exponentially forgetting obsolete evidence."
        ),
        (
            "What is the Li et al. (2010) Replay Methodology for offline evaluation?",
            "Direct online evaluation of untested policies risks revenue. The replay method steps through logged records (x_t, a_log, r_t) served uniformly at random. If the candidate policy's recommendation matches the logged ad, the interaction is accepted and evaluated; if not, it is ignored. This produces a provably unbiased estimate of live production CTR."
        )
    ]

    for q, a in viva_questions:
        with st.container(border=True):
            st.markdown(f"##### {q}")
            st.markdown(a)
