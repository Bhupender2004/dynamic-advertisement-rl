# Dynamic Advertisement Selection Using Upper Confidence Bound Reinforcement Learning

An academic Reinforcement Learning project modeling online digital advertisement placement as a **Multi-Armed Bandit (MAB)** problem using the classical **Upper Confidence Bound (UCB1)** algorithm.

---

## 1. Introduction

In modern online advertising systems (e.g., search engines, social media feeds, e-commerce banners), platforms face the challenge of serving the most relevant advertisement to maximize user engagement (Click-Through Rate / CTR). 

Traditional static A/B testing splits traffic equally across options for an extended duration, incurring significant revenue loss by serving underperforming ads. The **Multi-Armed Bandit (MAB)** framework provides an optimal alternative by dynamically adjusting exposure allocations in real-time based on empirical click feedback.

---

## 2. Problem Statement

How can an advertising platform maximize total clicks (cumulative reward) over $T$ rounds when the true Click-Through Rates of candidate advertisements are unknown *a priori*?

This introduces the fundamental **Exploration vs. Exploitation dilemma**:
* **Exploration:** Displaying less frequently visited advertisements to gather statistical evidence and refine CTR estimates.
* **Exploitation:** Displaying the currently highest-estimated advertisement to harvest immediate clicks.

---

## 3. Project Objectives

1. **Model Ad Selection as a Multi-Armed Bandit:** Formulate $K$ candidate advertisements as independent arms with stochastic Bernoulli payoff distributions.
2. **Implement UCB1 from Scratch:** Build the Upper Confidence Bound algorithm without relying on prepackaged black-box RL libraries.
3. **Simulate Online Feedback Loops:** Emulate real-time user click responses in a closed feedback environment.
4. **Demonstrate Sublinear Regret:** Show that the agent's cumulative regret grows logarithmically ($O(\ln T)$), proving asymptotic convergence to the optimal advertisement.
5. **Provide Interactive Dashboards:** Provide an interactive Streamlit application and Matplotlib visualizer for demonstration and viva defense.

---

## 4. Reinforcement Learning & Multi-Armed Bandit Mapping

The problem is structured as a non-associative, single-state Reinforcement Learning task (Multi-Armed Bandit):

| RL / MAB Concept | Advertising Domain Mapping | Project Implementation |
| :--- | :--- | :--- |
| **Agent** | Decision engine selecting which ad to display | `UCB1Agent` in `src/ucb_agent.py` |
| **Environment** | User audience & digital ad platform | `AdvertisementEnvironment` in `src/environment.py` |
| **Action ($a_t \in \{1 \dots K\}$)** | Serving a specific advertisement $i$ | `agent.select_ad()` |
| **State ($s_t$)** | Single persistent state (stateless bandit) | Implicit round index $t$ |
| **Reward ($r_t \in \{0, 1\}$)** | User click ($1$) or no click ($0$) | Sampled from $\text{Bernoulli}(p = \text{CTR}_i)$ |
| **Policy ($\pi$)** | UCB1 deterministic argmax rule | $\pi(t) = \arg\max_i \text{UCB}_i(t)$ |

---

## 5. Upper Confidence Bound (UCB1) Algorithm

The UCB1 algorithm (Auer et al., 2002) adopts the principle of **Optimism in the Face of Uncertainty**.

### Mathematical Formulation

For advertisement $i$ at round $t$:

$$\text{UCB}_i(t) = \hat{\mu}_i + c \sqrt{\frac{\ln(t)}{N_i}}$$

Where:
* $\hat{\mu}_i = \frac{R_i}{N_i}$: **Exploitation Term** (Empirical sample mean reward of ad $i$).
* $R_i$: Total accumulated clicks for ad $i$.
* $N_i$: Total times ad $i$ has been selected.
* $t$: Total simulation rounds across all advertisements ($t = \sum_{i=1}^K N_i$).
* $c$: **Exploration Coefficient** scaling confidence interval radius ($c=2.0$ or $c=\sqrt{2}$).
* $c \sqrt{\frac{\ln(t)}{N_i}}$: **Exploration Bonus** derived from Hoeffding's Inequality.

### Execution Steps
1. **Initialization Phase:** Pull each advertisement once ($t = 1 \dots K$) so that $N_i \ge 1$, avoiding division by zero.
2. **Selection Phase:** In round $t > K$, evaluate $\text{UCB}_i(t)$ for all candidate ads and pick $a_t = \arg\max_i \text{UCB}_i(t)$.
3. **Observation Phase:** Observe user interaction $r_t \sim \text{Bernoulli}(\text{CTR}_{a_t})$.
4. **Update Phase:** Increment $N_{a_t} \leftarrow N_{a_t} + 1$, $R_{a_t} \leftarrow R_{a_t} + r_t$, $\hat{\mu}_{a_t} \leftarrow \frac{R_{a_t}}{N_{a_t}}$, and $t \leftarrow t + 1$.

---

## 6. System Architecture

```mermaid
flowchart TD
    subgraph Environment ["Advertisement Environment (Ground Truth)"]
        HiddenCTR["Hidden CTRs: [p1, p2, ..., pK]"]
        BernoulliSampler["Bernoulli Sampler: r ~ Bernoulli(p_i)"]
    end

    subgraph Agent ["UCB1 Reinforcement Learning Agent"]
        StatTracker["Track Counts N_i & Total Clicks R_i"]
        UCBCalculator["Calculate UCB_i = mu_hat_i + c * sqrt(ln(t) / N_i)"]
        ArgMaxSelect["Select Action: a_t = argmax(UCB)"]
    end

    subgraph Evaluation ["Evaluation & Analytics"]
        Metrics["Log Cumulative Reward, Regret, CTR Estimates"]
        Plots["Generate Plots & Streamlit Visualizations"]
    end

    ArgMaxSelect -->|1. Display Selected Ad| BernoulliSampler
    BernoulliSampler -->|2. Return Binary Reward (0 or 1)| StatTracker
    StatTracker -->|3. Update Estimates| UCBCalculator
    UCBCalculator -->|4. Next Round Action Selection| ArgMaxSelect
    StatTracker -.->|5. Trajectory Logging| Metrics
    HiddenCTR -.->|Ground-truth baseline| Metrics
    Metrics --> Plots
```

---

## 7. Project Structure

```
dynamic-advertisement-rl/
│
├── src/
│   ├── __init__.py          # Package initialization
│   ├── environment.py       # AdvertisementEnvironment & Bernoulli reward generator
│   ├── ucb_agent.py         # Classical UCB1 Multi-Armed Bandit implementation
│   ├── simulation.py        # Coordinated simulation runner & time-series logger
│   └── evaluation.py        # Metrics calculations & Matplotlib visualization routines
│
├── dashboard/
│   └── app.py               # Streamlit interactive UI dashboard
│
├── data/
│   ├── .gitkeep
│   └── simulation_results.csv # Generated simulation trajectory logs
│
├── results/
│   ├── .gitkeep
│   ├── cumulative_reward.png
│   ├── cumulative_regret.png
│   ├── ad_selection_frequency.png
│   ├── ctr_comparison.png
│   └── combined_evaluation.png
│
├── tests/
│   ├── test_environment.py  # Environment unit tests
│   ├── test_ucb_agent.py    # Agent state & UCB score unit tests
│   └── test_simulation.py   # Simulation integrity & convergence unit tests
│
├── main.py                  # Command-line entry point
├── requirements.txt         # Dependencies
├── pytest.ini               # Test configuration
├── README.md                # Comprehensive project documentation
├── DEMO_GUIDE.md            # Review 1 presentation script & demo checklist
├── VIVA_QUESTIONS.md        # Academic viva preparation questions & answers
├── .gitignore               # Git ignore rules
└── LICENSE                  # MIT License
```

---

## 8. Evaluation Metrics

1. **Total Reward:** Total clicks harvested during the entire simulation:
   $$\text{Total Reward} = \sum_{t=1}^T r_t$$
2. **Average CTR:** Overall system click-through rate:
   $$\bar{r} = \frac{\text{Total Reward}}{T}$$
3. **Instantaneous Expected Regret:** The expected click deficit in round $t$ from not playing the optimal ad:
   $$\delta_t = \mu^* - \mu_{a_t}, \quad \text{where } \mu^* = \max_{i} \text{CTR}_i$$
4. **Cumulative Expected Regret:** Total cumulative penalty accumulated over $T$ rounds:
   $$\text{Regret}(T) = \sum_{t=1}^T (\mu^* - \mu_{a_t})$$
5. **Selection Frequency:** Proportion of total impressions allocated to each advertisement.
6. **Estimated CTR Convergence:** Convergence of $\hat{\mu}_i \to \text{CTR}_i$ as $N_i \to \infty$.

---

## 9. Installation & Setup

### Prerequisites
* Python 3.9+ (Python 3.11 recommended)
* Git

### Step-by-Step Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Bhupender2004/dynamic-advertisement-rl.git
   cd dynamic-advertisement-rl
   ```

2. **Create and activate a virtual environment:**
   * **Windows (PowerShell):**
     ```powershell
     python -m venv venv
     .\venv\Scripts\Activate.ps1
     ```
   * **macOS / Linux:**
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run unit tests:**
   ```bash
   pytest -v
   ```

---

## 10. How to Run

### Option A: Command-Line Simulation
Run the automated end-to-end benchmark script:
```bash
python main.py
```
This runs 10,000 rounds, prints an ASCII summary table, saves `data/simulation_results.csv`, and generates publication-quality plots in `results/`.

### Option B: Interactive Streamlit Dashboard
Launch the interactive web UI:
```bash
streamlit run dashboard/app.py
```
Navigate to `http://localhost:8501` to test custom CTR distributions, alter exploration parameters ($c$), and inspect real-time convergence curves.

---

## 11. Experimental Results & Observations

* **Optimal Arm Identification:** With benchmark CTRs $[0.05, 0.10, 0.03, 0.15, 0.08]$, the agent identifies **Ad 4** ($\text{CTR}=0.15$) as the winner and routes $>50\%$ of all impressions to it within $10,000$ rounds.
* **Sublinear Regret:** The cumulative regret curve flattens over time (logarithmic profile), validating theoretical guarantees.
* **Accurate CTR Estimation:** Sample means converge to ground-truth values without prior knowledge.

---

## 12. Current Project Limitations (Review 1 MVP)

1. **Stationary Rewards:** True CTRs remain constant over time; real-world ad fatigue or temporal trends are not yet modeled.
2. **Context-Free Bandits:** All simulated users share the same underlying click probabilities; individual user demographics and cookies are not factored into action selection.
3. **Simulated User Population:** Payoffs are generated via Bernoulli processes rather than live ad network logs.

---

## 13. Future Scope (Review 2 & Final Review)

* **Contextual Bandits (LinUCB / Disjoint LinUCB):** Personalizing advertisement recommendations using user feature vectors.
* **Algorithm Comparisons:** Benchmarking UCB1 against $\varepsilon$-Greedy, Decaying $\varepsilon$-Greedy, and Thompson Sampling (Bayesian Beta-Bernoulli).
* **Non-Stationary Bandits:** Implementing Discounted-UCB and Sliding-Window UCB for dynamically shifting CTR distributions.
* **Real-World Ad Datasets:** Validating algorithms against public datasets (e.g., Avazu, Criteo CTR dataset).

---

## 14. License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
