# Review 1 Demonstration Guide & Script

**Project Title:** Dynamic Advertisement Selection Using Upper Confidence Bound Reinforcement Learning  
**Milestone:** Review 1 (~50% Completion)  
**Estimated Demo Duration:** 5 to 8 Minutes  

---

## 🎯 Demonstration Objective
Demonstrate a clean, mathematically rigorous, and fully functional Multi-Armed Bandit (MAB) Reinforcement Learning system using the UCB1 algorithm to solve dynamic advertisement allocation under uncertainty.

---

## 📋 Step-by-Step Presentation Script

### 1. Introduction & Problem Statement (1 Minute)
> *"Good morning/afternoon, Professors. Today I am presenting Review 1 for our project: Dynamic Advertisement Selection Using Upper Confidence Bound Reinforcement Learning.*
>
> *In online digital advertising, platforms have multiple candidate ads but do not know which ones users are most likely to click. Traditional A/B testing wastes revenue on poor ads for weeks. Our project formulates this as a Multi-Armed Bandit problem where an RL agent dynamically explores candidate ads and exploits the highest-performing ones in real-time."*

---

### 2. Reinforcement Learning & MAB Formulation (1 Minute)
> *"We model the advertising environment as a non-associative Reinforcement Learning problem:*
> - *The **Agent** is our UCB1 algorithm.*
> - *The **Actions** are the candidate advertisements (Arms).*
> - *The **Reward** is binary (1 for a user click, 0 for no click) generated from a Bernoulli distribution.*
> - *The **Objective** is to maximize total clicks (cumulative reward) and minimize cumulative regret over time."*

---

### 3. Theoretical Principle & The UCB1 Formula (1.5 Minutes)
> *"The agent balances Exploration vs. Exploitation using the Upper Confidence Bound formula derived from Hoeffding's Inequality:*
>
> $$\text{UCB}_i(t) = \hat{\mu}_i + c \sqrt{\frac{\ln(t)}{N_i}}$$
>
> - *$\hat{\mu}_i$ is the **Exploitation term**—the current empirical average CTR of advertisement $i$.*
> - *$c \sqrt{\frac{\ln(t)}{N_i}}$ is the **Exploration bonus**—representing uncertainty. When an ad hasn't been shown often, $N_i$ is small, inflating the bound and forcing the agent to explore it.*
> - *As time $t$ grows, $\ln(t)$ prevents stale arms from being permanently ignored while allowing the agent to focus majority traffic on the winner."*

---

### 4. Code Architecture & Execution (1.5 Minutes)
> *"Our project is structured into clean, modular Python components without reliance on high-level black-box bandit libraries:*
> - `src/environment.py`: *Simulates stochastic user response with hidden CTRs.*
> - `src/ucb_agent.py`: *Implements the UCB1 algorithm from scratch.*
> - `src/simulation.py`: *Coordinates the interaction loop and logs time-series trajectories.*
> - `src/evaluation.py`: *Computes mathematical metrics and generates plots.*
>
> *Let's first run our command-line pipeline: `python main.py`."*

*(Run `python main.py` in the terminal and point to the results)*

> *"Notice how out of 10,000 rounds:*
> - *Ad 4 (with the true hidden CTR of 15%) received over 51% of total traffic.*
> - *The estimated CTRs for all 5 ads closely converged to their ground truth values.*
> - *The system generated high-resolution plots in the `results/` folder."*

---

### 5. Interactive Dashboard Demonstration (2 Minutes)
> *"Now let's launch our interactive Streamlit application: `streamlit run dashboard/app.py`."*

*(Open browser at `http://localhost:8501`)*

**Show these key dashboard sections:**
1. **Sidebar Controls:** Demonstrate changing simulation rounds or exploration parameter $c$.
2. **Key Performance Indicators:** Point out Total Clicks, Overall Average CTR, and the Agent's identified winning ad.
3. **Performance Matrix Table:** Point out how the UCB empirical CTR estimate is within decimal points of the hidden environment CTR.
4. **Learning Dynamics Graphs:**
   - **Cumulative Reward:** Point to the upward linear trajectory as clicks accumulate.
   - **Cumulative Regret:** Point out that the curve flattens over time (logarithmic regret), proving the agent learns and stops making sub-optimal choices.
5. **Selection Frequency:** Show the bar chart highlighting how Ad 4 dominates impressions.

---

### 6. Limitations & Future Scope (50% $\to$ 100% Roadmap) (1 Minute)
> *"For Review 1, we have successfully completed 50% of the project encompassing the core theoretical and empirical MAB pipeline.*
>
> *For Review 2 and Final Submission, our roadmap includes:*
> 1. *Benchmarking UCB against **Thompson Sampling (Bayesian RL)** and **$\varepsilon$-Greedy** policies.*
> 2. *Upgrading to **Contextual Bandits (LinUCB)** where user feature vectors (demographics, device type) dynamically condition ad selection.*
> 3. *Evaluating against non-stationary CTR shifts (ad fatigue).*
> 4. *Testing on real-world benchmark ad datasets (e.g., Criteo / Avazu)."*

---

## ⚡ Quick Reference Numbers (Default Benchmark)

| Metric | Benchmark Value |
| :--- | :--- |
| Candidate Ads | 5 Ads (`Ad 1` to `Ad 5`) |
| True Hidden CTRs | $[0.05, 0.10, 0.03, 0.15, 0.08]$ |
| Optimal Ad | `Ad 4` (15% CTR) |
| Total Rounds | 10,000 |
| Total Clicks Achieved | ~1,120 – 1,150 clicks |
| Ad 4 Selection Share | >50% of total impressions |
| Regret Growth Profile | Sublinear / Logarithmic ($O(\ln T)$) |
