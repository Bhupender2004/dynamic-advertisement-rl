"""
Streamlit Master Academic Dashboard for Dynamic Advertisement Selection.

Comprehensive Reinforcement Learning showcase:
- Tab 1: 🏆 UCB1 Deep Dive & Learning Convergence
- Tab 2: ⚔️ Multi-Algorithm Benchmark Arena (UCB1, Thompson Sampling, ε-Greedy, Random)
- Tab 3: 🎯 Contextual Personalization (Disjoint LinUCB with Live User Persona Simulator)
- Tab 4: 🔄 Non-Stationary Dynamics & Ad Fatigue Adaptation
- Tab 5: 📊 Unbiased Offline Policy Evaluation (Replay Methodology)
- Tab 6: 📚 Academic Viva Defense & Theoretical Compendium
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

# Page configuration
st.set_page_config(
    page_title="Dynamic Ad Selection with RL & Bandits",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for High-Contrast, Crystal-Clear Academic Aesthetic
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f8fafc;
        color: #0f172a;
    }
    .main-title {
        font-size: 2.1rem;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 0.2rem;
        letter-spacing: -0.02em;
    }
    .sub-title {
        font-size: 1.0rem;
        font-weight: 500;
        color: #475569;
        margin-bottom: 1.2rem;
    }
    h2, h3, .stHeading {
        color: #0f172a !important;
        font-weight: 700 !important;
    }
    .metric-card {
        background: #ffffff;
        border: 1px solid #cbd5e1;
        border-radius: 10px;
        padding: 1.0rem 0.8rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        margin-bottom: 0.8rem;
    }
    .metric-val {
        font-size: 1.6rem;
        font-weight: 800;
        color: #1e3a8a;
        margin-top: 0.1rem;
    }
    .metric-lbl {
        font-size: 0.75rem;
        font-weight: 700;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .badge-win {
        display: inline-block;
        background-color: #dcfce7;
        color: #166534;
        font-size: 0.75rem;
        font-weight: 700;
        padding: 0.15rem 0.4rem;
        border-radius: 4px;
        margin-left: 0.3rem;
    }
    .theory-box {
        background-color: #ffffff;
        border: 1px solid #bfdbfe;
        border-left: 4px solid #2563eb;
        padding: 1.0rem 1.2rem;
        border-radius: 8px;
        margin: 1.0rem 0;
        box-shadow: 0 1px 4px rgba(37, 99, 235, 0.05);
    }
    .qa-box {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.8rem;
    }
    .qa-q {
        font-weight: 700;
        color: #1e3a8a;
        font-size: 1.0rem;
        margin-bottom: 0.4rem;
    }
    .qa-a {
        color: #334155;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header
st.markdown('<div class="main-title">Dynamic Advertisement Selection Using Reinforcement Learning</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Multi-Armed Bandit & Contextual Reinforcement Learning Platform (Final Capstone Submission)</div>', unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🏆 Single-Agent UCB1",
    "⚔️ Algorithm Arena",
    "🎯 Contextual Personalization (LinUCB)",
    "🔄 Non-Stationary Dynamics",
    "📊 Offline Replay Evaluation",
    "📚 Theory & Viva Defense",
])


# ==============================================================================
# TAB 1: SINGLE-AGENT UCB1 DEEP DIVE
# ==============================================================================
with tab1:
    st.subheader("1. Classical Upper Confidence Bound (UCB1) Simulation")
    st.caption("Investigate exploration vs. exploitation trade-offs, confidence interval shrinkage, and logarithmic regret.")

    col_cfg1, col_cfg2, col_cfg3 = st.columns([1, 1, 1])
    with col_cfg1:
        num_rounds_ucb = st.slider("Simulation Rounds (t)", min_value=1000, max_value=30000, value=10000, step=1000, key="ucb_rounds")
    with col_cfg2:
        exploration_c = st.slider("Exploration Coefficient (c)", min_value=0.1, max_value=4.0, value=2.0, step=0.1, key="ucb_c")
    with col_cfg3:
        ucb_seed = st.number_input("Random Seed", value=42, step=1, key="ucb_seed")

    with st.expander("⚙️ Configure Hidden Ground Truth CTRs (Environment)", expanded=False):
        col_c1, col_c2, col_c3, col_c4, col_c5 = st.columns(5)
        ctr1 = col_c1.slider("Ad 1 CTR", 0.01, 0.40, 0.05, 0.01)
        ctr2 = col_c2.slider("Ad 2 CTR", 0.01, 0.40, 0.10, 0.01)
        ctr3 = col_c3.slider("Ad 3 CTR", 0.01, 0.40, 0.03, 0.01)
        ctr4 = col_c4.slider("Ad 4 CTR", 0.01, 0.40, 0.15, 0.01)
        ctr5 = col_c5.slider("Ad 5 CTR", 0.01, 0.40, 0.08, 0.01)

    true_ctrs_tab1 = [ctr1, ctr2, ctr3, ctr4, ctr5]
    ad_names_tab1 = ["Ad 1", "Ad 2", "Ad 3", "Ad 4", "Ad 5"]

    @st.cache_data(show_spinner=False)
    def cached_ucb1_sim(ctrs, rounds, c, seed):
        env = AdvertisementEnvironment(true_ctrs=ctrs, ad_names=ad_names_tab1, seed=seed)
        agent = UCB1Agent(num_ads=len(ctrs), c=c)
        df = run_simulation(env=env, agent=agent, num_rounds=rounds, seed=seed)
        metrics = calculate_metrics(df, env, agent)
        return df, metrics

    df_ucb, metrics_ucb = cached_ucb1_sim(true_ctrs_tab1, num_rounds_ucb, exploration_c, ucb_seed)

    # KPI Scorecards
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f'<div class="metric-card"><div class="metric-lbl">Total Clicks Harvested</div><div class="metric-val">{metrics_ucb["total_reward"]:,}</div></div>', unsafe_allow_html=True)
    with k2:
        st.markdown(f'<div class="metric-card"><div class="metric-lbl">Average System CTR</div><div class="metric-val">{metrics_ucb["average_reward"]*100:.2f}%</div></div>', unsafe_allow_html=True)
    with k3:
        st.markdown(f'<div class="metric-card"><div class="metric-lbl">Optimal Ad (Ground Truth)</div><div class="metric-val">{metrics_ucb["best_ad_name"]} ({metrics_ucb["max_true_ctr"]*100:.0f}%)</div></div>', unsafe_allow_html=True)
    with k4:
        is_match = (metrics_ucb["agent_favorite_name"] == metrics_ucb["best_ad_name"])
        badge = '<span class="badge-win">Optimal Winner</span>' if is_match else ''
        st.markdown(f'<div class="metric-card"><div class="metric-lbl">Agent Favorite Choice</div><div class="metric-val">{metrics_ucb["agent_favorite_name"]} {badge}</div></div>', unsafe_allow_html=True)

    # Detailed Summary Table
    st.dataframe(metrics_ucb["ad_summary_df"], width="stretch", hide_index=True)

    # Visualizations
    pcol1, pcol2 = st.columns(2)
    with pcol1:
        fig1, ax1 = plt.subplots(figsize=(6, 3.8))
        ax1.plot(df_ucb["round"], df_ucb["cumulative_reward"], color="#1f77b4", linewidth=2.2, label="Cumulative Clicks")
        ax1.set_title("Cumulative Clicks Over Time", fontweight="bold", pad=8)
        ax1.set_xlabel("Round (t)")
        ax1.set_ylabel("Clicks")
        ax1.grid(True, linestyle="--", alpha=0.6)
        ax1.legend()
        st.pyplot(fig1, width="stretch")
        plt.close(fig1)

    with pcol2:
        fig2, ax2 = plt.subplots(figsize=(6, 3.8))
        ax2.plot(df_ucb["round"], df_ucb["cumulative_expected_regret"], color="#dc2626", linewidth=2.2, label="Cumulative Regret")
        ax2.set_title("Cumulative Regret (Sublinear Logarithmic Growth)", fontweight="bold", pad=8)
        ax2.set_xlabel("Round (t)")
        ax2.set_ylabel("Regret")
        ax2.grid(True, linestyle="--", alpha=0.6)
        ax2.legend()
        st.pyplot(fig2, width="stretch")
        plt.close(fig2)

    pcol3, pcol4 = st.columns(2)
    with pcol3:
        fig3, ax3 = plt.subplots(figsize=(6, 3.8))
        best_idx = np.argmax(true_ctrs_tab1)
        bar_colors = ["#16a34a" if i == best_idx else "#3b82f6" for i in range(len(true_ctrs_tab1))]
        counts = metrics_ucb["selection_counts"]
        bars = ax3.bar(ad_names_tab1, counts, color=bar_colors, edgecolor="black", alpha=0.85)
        for b in bars:
            ax3.annotate(f"{int(b.get_height()):,}", xy=(b.get_x() + b.get_width() / 2, b.get_height()), xytext=(0, 3), textcoords="offset points", ha="center", fontweight="bold", fontsize=8)
        ax3.set_title("Ad Selection Frequency (Exploitation Winner)", fontweight="bold", pad=8)
        ax3.set_ylabel("Impressions")
        ax3.grid(True, axis="y", linestyle="--", alpha=0.6)
        st.pyplot(fig3, width="stretch")
        plt.close(fig3)

    with pcol4:
        fig4, ax4 = plt.subplots(figsize=(6, 3.8))
        x_idx = np.arange(len(true_ctrs_tab1))
        w = 0.35
        ax4.bar(x_idx - w/2, true_ctrs_tab1, w, label="True CTR", color="#4f46e5", alpha=0.85)
        ax4.bar(x_idx + w/2, metrics_ucb["estimated_ctrs"], w, label="Estimated CTR (UCB)", color="#f59e0b", alpha=0.85)
        ax4.set_xticks(x_idx)
        ax4.set_xticklabels(ad_names_tab1)
        ax4.set_title("True CTR vs. Estimated CTR Convergence", fontweight="bold", pad=8)
        ax4.set_ylabel("CTR")
        ax4.grid(True, axis="y", linestyle="--", alpha=0.6)
        ax4.legend()
        st.pyplot(fig4, width="stretch")
        plt.close(fig4)


# ==============================================================================
# TAB 2: MULTI-ALGORITHM BENCHMARK ARENA
# ==============================================================================
with tab2:
    st.subheader("2. Multi-Algorithm Benchmark Arena")
    st.caption("Direct head-to-head empirical comparison of 5 core policies under identical environmental seeds.")

    col_arena1, col_arena2 = st.columns([1, 1])
    with col_arena1:
        rounds_arena = st.slider("Benchmark Rounds", min_value=1000, max_value=25000, value=10000, step=1000, key="arena_rounds")
    with col_arena2:
        seed_arena = st.number_input("Benchmark Seed", value=42, step=1, key="arena_seed")

    @st.cache_data(show_spinner=False)
    def cached_arena_sim(rounds, seed):
        true_ctrs = [0.05, 0.10, 0.03, 0.15, 0.08]
        agents = {
            "UCB1 (c=2.0)": UCB1Agent(num_ads=5, c=2.0),
            "Thompson Sampling": ThompsonSamplingAgent(num_ads=5, seed=seed),
            "Epsilon-Greedy (decay)": EpsilonGreedyAgent(num_ads=5, epsilon=0.5, decay=True, seed=seed),
            "Epsilon-Greedy (eps=0.1)": EpsilonGreedyAgent(num_ads=5, epsilon=0.1, decay=False, seed=seed),
            "Random / Static A/B": RandomAgent(num_ads=5, seed=seed),
        }
        res = run_multi_agent_simulation(agents=agents, true_ctrs=true_ctrs, num_rounds=rounds, seed=seed)
        return res

    with st.spinner("Running Multi-Algorithm Benchmark Arena..."):
        arena_results = cached_arena_sim(rounds_arena, seed_arena)

    # Leaderboard calculation
    arena_lb = []
    for name, df in arena_results.items():
        clicks = int(df["cumulative_reward"].iloc[-1])
        regret = float(df["cumulative_expected_regret"].iloc[-1])
        arena_lb.append({
            "Algorithm Policy": name,
            "Total Clicks": clicks,
            "Effective System CTR": f"{clicks / len(df) * 100:.2f}%",
            "Cumulative Regret": round(regret, 2),
            "Theoretical Regret Bound": "O(ln T)" if "UCB" in name or "Thompson" in name or "decay" in name else "O(T) [Linear Loss]",
        })

    df_arena_lb = pd.DataFrame(arena_lb).sort_values(by="Total Clicks", ascending=False)

    best_algo = df_arena_lb.iloc[0]["Algorithm Policy"]
    best_clicks = df_arena_lb.iloc[0]["Total Clicks"]
    random_clicks = df_arena_lb[df_arena_lb["Algorithm Policy"].str.contains("Random")]["Total Clicks"].values[0]
    improvement = ((best_clicks - random_clicks) / random_clicks) * 100

    ak1, ak2, ak3 = st.columns(3)
    with ak1:
        st.markdown(f'<div class="metric-card"><div class="metric-lbl">Arena Winner</div><div class="metric-val">{best_algo}</div></div>', unsafe_allow_html=True)
    with ak2:
        st.markdown(f'<div class="metric-card"><div class="metric-lbl">Max Clicks Harvested</div><div class="metric-val">{best_clicks:,}</div></div>', unsafe_allow_html=True)
    with ak3:
        st.markdown(f'<div class="metric-card"><div class="metric-lbl">Lift vs Static A/B</div><div class="metric-val">+{improvement:.1f}% Clicks</div></div>', unsafe_allow_html=True)

    st.dataframe(df_arena_lb, width="stretch", hide_index=True)

    # Arena Comparison Graphs
    acol1, acol2 = st.columns(2)
    with acol1:
        fig_r, ax_r = plt.subplots(figsize=(6, 3.8))
        for name, df in arena_results.items():
            ax_r.plot(df["round"], df["cumulative_expected_regret"], label=name, linewidth=2)
        ax_r.set_title("Cumulative Regret Comparison", fontweight="bold", pad=8)
        ax_r.set_xlabel("Round (t)")
        ax_r.set_ylabel("Regret")
        ax_r.grid(True, linestyle="--", alpha=0.6)
        ax_r.legend(fontsize=8)
        st.pyplot(fig_r, width="stretch")
        plt.close(fig_r)

    with acol2:
        fig_c, ax_c = plt.subplots(figsize=(6, 3.8))
        for name, df in arena_results.items():
            ax_c.plot(df["round"], df["cumulative_reward"], label=name, linewidth=2)
        ax_c.set_title("Cumulative Clicks Harvested", fontweight="bold", pad=8)
        ax_c.set_xlabel("Round (t)")
        ax_c.set_ylabel("Total Clicks")
        ax_c.grid(True, linestyle="--", alpha=0.6)
        ax_c.legend(fontsize=8)
        st.pyplot(fig_c, width="stretch")
        plt.close(fig_c)


# ==============================================================================
# TAB 3: CONTEXTUAL PERSONALIZATION (LINUCB)
# ==============================================================================
with tab3:
    st.subheader("3. Contextual Bandits: Disjoint LinUCB Personalization")
    st.caption("Move beyond context-free bandits by personalizing ad recommendations based on user feature vectors x_t.")

    c_col1, c_col2 = st.columns([1, 1.2])

    with c_col1:
        st.markdown("#### 👤 Live User Persona Simulator")
        st.caption("Configure an incoming user's traits and see which ad LinUCB serves:")
        u_mobile = st.checkbox("📱 Mobile Device User (vs. Desktop)", value=True)
        u_young = st.checkbox("👶 Age Under 30 (vs. Older)", value=True)
        u_evening = st.checkbox("🌙 Evening / Weekend Browsing (vs. Work Hours)", value=False)
        u_tech = st.checkbox("🎮 High Tech & Gaming Affinity", value=True)

        user_context_vec = np.array([
            1.0,  # Bias
            1.0 if u_mobile else 0.0,
            1.0 if u_young else 0.0,
            1.0 if u_evening else 0.0,
            1.0 if u_tech else 0.0,
        ], dtype=np.float64)

        alpha_linucb = st.slider("LinUCB Confidence Parameter (α)", 0.1, 3.0, 1.0, 0.1)

    @st.cache_data(show_spinner=False)
    def cached_contextual_sim(rounds, alpha, seed):
        env = ContextualAdvertisementEnvironment(num_ads=4, context_dim=5, seed=seed)
        agent = LinUCBAgent(num_ads=4, context_dim=5, alpha=alpha)
        df = run_contextual_simulation(env, agent, num_rounds=rounds, seed=seed)
        thetas = agent.get_estimated_thetas()
        return df, thetas, env, agent

    df_ctx, learned_thetas, c_env, c_agent = cached_contextual_sim(10000, alpha_linucb, 42)

    with c_col2:
        st.markdown("#### 🎯 LinUCB Recommendation")
        recommended_ad_idx = c_agent.select_ad(user_context_vec)
        rec_name = c_env.ad_names[recommended_ad_idx]
        probs = c_env.get_ad_click_probabilities(user_context_vec)

        st.success(f"**Recommended Advertisement:** **{rec_name}**")
        st.markdown(f"**True Underlying Click Probability:** **{probs[recommended_ad_idx]*100:.1f}%**")

        st.markdown("**Predicted User Affinity Breakdown:**")
        affinity_df = pd.DataFrame({
            "Candidate Ad": c_env.ad_names,
            "Expected CTR": [f"{p*100:.1f}%" for p in probs],
            "Recommended": ["⭐ YES" if i == recommended_ad_idx else "No" for i in range(len(probs))],
        })
        st.dataframe(affinity_df, width="stretch", hide_index=True)

    st.markdown("---")
    st.markdown("#### 📈 10,000 Round Personalized Simulation Trajectory")
    cx_col1, cx_col2 = st.columns(2)
    with cx_col1:
        fig_c1, ax_c1 = plt.subplots(figsize=(6, 3.6))
        ax_c1.plot(df_ctx["round"], df_ctx["cumulative_regret"], color="#8b5cf6", linewidth=2.2)
        ax_c1.set_title("LinUCB Cumulative Contextual Regret", fontweight="bold", pad=8)
        ax_c1.set_xlabel("Round (t)")
        ax_c1.set_ylabel("Regret")
        ax_c1.grid(True, linestyle="--", alpha=0.6)
        st.pyplot(fig_c1, width="stretch")
        plt.close(fig_c1)

    with cx_col2:
        fig_c2, ax_c2 = plt.subplots(figsize=(6, 3.6))
        alloc_counts = df_ctx["ad_name"].value_counts()
        ax_c2.bar(alloc_counts.index, alloc_counts.values, color="#6366f1", edgecolor="black", alpha=0.85)
        for i, val in enumerate(alloc_counts.values):
            ax_c2.annotate(f"{val:,}", xy=(i, val), xytext=(0, 3), textcoords="offset points", ha="center", fontweight="bold")
        ax_c2.set_title("LinUCB Personalized Traffic Distribution", fontweight="bold", pad=8)
        ax_c2.set_ylabel("Impressions Served")
        ax_c2.tick_params(axis="x", rotation=15)
        ax_c2.grid(True, axis="y", linestyle="--", alpha=0.6)
        st.pyplot(fig_c2, width="stretch")
        plt.close(fig_c2)


# ==============================================================================
# TAB 4: NON-STATIONARY DYNAMICS & AD FATIGUE
# ==============================================================================
with tab4:
    st.subheader("4. Non-Stationary Environments & Ad Fatigue Adaptation")
    st.caption("Demonstrate how standard stationary UCB1 fails when ad performance shifts, and how Discounted-UCB adapts.")

    ns_col1, ns_col2 = st.columns([1, 1])
    with ns_col1:
        shift_point = st.slider("Ad Fatigue Shift Round", min_value=2000, max_value=8000, value=5000, step=500)
    with ns_col2:
        discount_gamma = st.slider("Discount Factor (γ)", min_value=0.990, max_value=0.9999, value=0.998, step=0.0005, format="%.4f")

    st.info(f"**Regime Shift Details:** Before round {shift_point:,}, Ad 4 is optimal (CTR = 15%). At round {shift_point:,}, Ad 4 suffers severe ad fatigue (CTR drops to 2%), and Ad 2 surges to become the new winner (CTR = 18%).")

    @st.cache_data(show_spinner=False)
    def cached_non_stationary_sim(shift, gamma, seed=42):
        env = NonStationaryEnvironment(
            initial_ctrs=[0.05, 0.10, 0.03, 0.15, 0.08],
            change_points={shift: [0.05, 0.18, 0.03, 0.02, 0.08]},
            seed=seed,
        )
        agents = {
            "Stationary UCB1": UCB1Agent(num_ads=5, c=1.0),
            f"Discounted-UCB (γ={gamma})": DiscountedUCBAgent(num_ads=5, gamma=gamma, c=0.2),
            "Sliding-Window UCB (W=1500)": SlidingWindowUCBAgent(num_ads=5, window_size=1500, c=0.2),
        }
        return run_non_stationary_simulation(env, agents, num_rounds=10000, seed=seed)

    ns_results = cached_non_stationary_sim(shift_point, discount_gamma)

    fig_ns, ax_ns = plt.subplots(figsize=(10, 4.2))
    colors_ns = {"Stationary UCB1": "#dc2626", f"Discounted-UCB (γ={discount_gamma})": "#16a34a", "Sliding-Window UCB (W=1500)": "#2563eb"}
    for name, df in ns_results.items():
        ax_ns.plot(df["round"], df["cumulative_regret"], label=name, color=colors_ns.get(name, "#333333"), linewidth=2.2)

    ax_ns.axvline(x=shift_point, color="#0f172a", linestyle=":", linewidth=2, label=f"Fatigue Shift (t={shift_point:,})")
    ax_ns.set_title("Regret Over Time Under Non-Stationary Ad Fatigue", fontweight="bold", pad=10)
    ax_ns.set_xlabel("Simulation Round (t)")
    ax_ns.set_ylabel("Cumulative Regret")
    ax_ns.grid(True, linestyle="--", alpha=0.6)
    ax_ns.legend()
    st.pyplot(fig_ns, width="stretch")
    plt.close(fig_ns)


# ==============================================================================
# TAB 5: OFFLINE POLICY EVALUATION (REPLAY METHOD)
# ==============================================================================
with tab5:
    st.subheader("5. Unbiased Offline Policy Evaluation (Replay Methodology)")
    st.caption("Statistically validate new algorithms on logged historical ad impressions without risking live revenue (Li et al., WWW 2010).")

    dataset_path = "data/real_ad_clicks.csv"
    if not os.path.exists(dataset_path):
        generate_sample_ad_dataset(output_path=dataset_path, num_samples=10000, seed=42)

    df_logs = pd.read_csv(dataset_path)

    st.markdown(f"**Loaded Offline Click Log:** `{len(df_logs):,}` historical impression trials with user demographics and click feedback.")

    @st.cache_data(show_spinner=False)
    def cached_offline_eval():
        u_agent = UCB1Agent(num_ads=4, c=1.5)
        t_agent = ThompsonSamplingAgent(num_ads=4, seed=42)
        l_agent = LinUCBAgent(num_ads=4, context_dim=5, alpha=1.0)

        ev_u = evaluate_policy_offline_replay(u_agent, df_logs, is_contextual=False)
        ev_t = evaluate_policy_offline_replay(t_agent, df_logs, is_contextual=False)
        ev_l = evaluate_policy_offline_replay(l_agent, df_logs, is_contextual=True)
        return ev_u, ev_t, ev_l

    ev_u, ev_t, ev_l = cached_offline_eval()

    off_summary = pd.DataFrame([
        {"Evaluated Policy": "UCB1 (MAB)", "Accepted Replay Impressions": f"{ev_u['accepted_rounds']:,}", "Clicks Harvested": ev_u["total_clicks"], "Effective Offline CTR": f"{ev_u['effective_ctr']*100:.2f}%"},
        {"Evaluated Policy": "Thompson Sampling (MAB)", "Accepted Replay Impressions": f"{ev_t['accepted_rounds']:,}", "Clicks Harvested": ev_t["total_clicks"], "Effective Offline CTR": f"{ev_t['effective_ctr']*100:.2f}%"},
        {"Evaluated Policy": "Contextual LinUCB", "Accepted Replay Impressions": f"{ev_l['accepted_rounds']:,}", "Clicks Harvested": ev_l["total_clicks"], "Effective Offline CTR": f"{ev_l['effective_ctr']*100:.2f}%"},
    ])
    st.dataframe(off_summary, width="stretch", hide_index=True)

    fig_off, ax_off = plt.subplots(figsize=(9, 4))
    ax_off.plot(ev_u["running_ctr_history"], label="UCB1 Effective CTR", color="#1f77b4", linewidth=2)
    ax_off.plot(ev_t["running_ctr_history"], label="Thompson Sampling Effective CTR", color="#2ca02c", linewidth=2)
    ax_off.plot(ev_l["running_ctr_history"], label="Contextual LinUCB Effective CTR", color="#8b5cf6", linewidth=2)
    ax_off.set_title("Offline Replay Effective CTR Convergence", fontweight="bold", pad=8)
    ax_off.set_xlabel("Accepted Trials")
    ax_off.set_ylabel("Running CTR")
    ax_off.grid(True, linestyle="--", alpha=0.6)
    ax_off.legend()
    st.pyplot(fig_off, width="stretch")
    plt.close(fig_off)


# ==============================================================================
# TAB 6: ACADEMIC VIVA & THEORY DEFENSE
# ==============================================================================
with tab6:
    st.subheader("6. Academic Viva Defense & Theoretical Foundations")
    st.caption("Preparation guide for academic professors, reviewers, and oral viva examination.")

    viva_qas = [
        ("Q1: What is a Multi-Armed Bandit (MAB) problem and how does it relate to RL?",
         "A Multi-Armed Bandit is a foundational simplified formulation of Reinforcement Learning where an agent faces K choices (arms), each with an unknown payoff distribution. Unlike standard MDPs, MAB is non-associative and stateless (actions do not change future environmental states). The objective is to maximize cumulative clicks over time."),
        ("Q2: Explain the UCB1 mathematical formula and the Hoeffding bound.",
         "UCB_i(t) = hat{mu}_i + c * sqrt(ln(t) / N_i). The first term represents exploitation (sample mean CTR). The second term represents exploration bonus derived from Hoeffding's Inequality. As t increases, ln(t) grows slowly, preventing stale arms from being ignored while shrinking the bonus for frequently pulled arms."),
        ("Q3: How does UCB differ from Thompson Sampling?",
         "UCB is a deterministic frequentist algorithm that constructs optimistic upper confidence bounds. Thompson Sampling is a Bayesian probabilistic algorithm that maintains Beta(alpha, beta) posterior distributions over CTRs and samples from them at each step, matching the Lai-Robbins theoretical lower bound."),
        ("Q4: Why does static A/B testing fail compared to MAB?",
         "Static A/B testing allocates fixed traffic (e.g., 20% to each of 5 ads) regardless of performance, incurring linear regret O(T). MAB dynamically shifts traffic to the winning ad as evidence accumulates, achieving logarithmic regret O(ln T) and minimizing revenue deficit."),
        ("Q5: What are Contextual Bandits and why use LinUCB?",
         "Context-free MAB assumes all users share identical click probabilities. In digital advertising, user demographics (mobile, age, time of day) heavily impact clicks. LinUCB estimates user-ad affinity vectors theta_a via ridge regression, enabling personalized ad recommendations."),
        ("Q6: How does the system handle non-stationarity and ad fatigue?",
         "In stationary bandits, past observations have permanent weight. In real advertising, ad fatigue degrades performance over time. Discounted-UCB (D-UCB) applies geometric decay gamma^(t-s) to past observations, exponentially forgetting stale history and adapting immediately when CTRs shift."),
        ("Q7: What is the Li et al. (2010) Replay Methodology for offline evaluation?",
         "Deploying untested algorithms directly to live users risks financial loss. The replay methodology passes historical logged clicks through the agent: if the agent picks the same ad as logged in history, the event is accepted and evaluated; otherwise it is rejected. This provides a provably unbiased evaluation of online policy performance."),
    ]

    for q, a in viva_qas:
        st.markdown(
            f"""
            <div class="qa-box">
                <div class="qa-q">{q}</div>
                <div class="qa-a">{a}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
