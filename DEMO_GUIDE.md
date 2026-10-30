# Final Capstone Project Demonstration Guide & Presentation Script

**Project Title:** Dynamic Advertisement Selection Using Multi-Armed Bandit & Contextual Reinforcement Learning  
**Milestone:** Final Review & Capstone Defense (100% Full Project Completion)  
**Estimated Demo Duration:** 8 to 12 Minutes  
**Live UI:** `streamlit run dashboard/app.py`  
**Master CLI:** `python main.py`  
**Test Suite:** `pytest -v` (36 Passed)  

---

## 🎯 Demonstration Objective
Demonstrate an end-to-end, mathematically rigorous Reinforcement Learning framework built entirely from scratch to solve online digital advertisement placement under statistical uncertainty. 

The demonstration highlights:
1. **Classical MAB:** UCB1, Thompson Sampling, $\varepsilon$-Greedy, and Static Random A/B Testing.
2. **Contextual Personalization:** Disjoint LinUCB optimizing for multi-dimensional user personas.
3. **Market Volatility & Ad Fatigue:** Non-stationary adaptation using Discounted-UCB and Sliding-Window UCB.
4. **Offline Evaluation:** Unbiased policy replay on historical impression logs (Li et al., WWW 2010).

---

## 📋 Step-by-Step Presentation Script

### 1. Opening & Problem Statement (1 Minute)
> *"Good morning/afternoon, Professors and Reviewers. Today I am presenting our completed Capstone project: Dynamic Advertisement Selection Using Multi-Armed Bandit & Contextual Reinforcement Learning.*
>
> *In digital advertising, platforms have multiple ads to serve but cannot know user click-through rates (CTRs) in advance. Traditional static A/B testing splits traffic equally, incurring massive opportunity loss on underperforming ads. Our project solves this through an adaptive Reinforcement Learning framework that optimizes the exploration-exploitation trade-off in real time."*

---

### 2. Theoretical Foundations & Algorithmic Arsenal (2 Minutes)
> *"We formulate the problem across three progressive reinforcement learning tiers:*
>
> 1. *Tier 1: Classical Multi-Armed Bandit (MAB):*
>    - **UCB1:** Uses Hoeffding's Inequality to compute $\text{UCB}_i(t) = \hat{\mu}_i + c \sqrt{\frac{\ln(t)}{N_i}}$, balancing exploitation with statistical uncertainty.
>    - **Thompson Sampling:** A Bayesian approach maintaining $\text{Beta}(\alpha, \beta)$ conjugate posteriors over each arm's CTR.
>    - **$\varepsilon$-Greedy:** Both fixed and decaying exploration schedules.
>
> 2. *Tier 2: Contextual Bandits (Disjoint LinUCB):*
>    - Real users have distinct traits (device, age, time of day). We implement LinUCB using ridge regression covariance matrices $A_a$ and bias vectors $b_a$ to predict personalized click probabilities: $p_{t, a} = x_t^T \hat{\theta}_a + \alpha \sqrt{x_t^T A_a^{-1} x_t}$.
>
> 3. *Tier 3: Non-Stationary Dynamics (Ad Fatigue):*
>    - Real ads degrade over time. We implement Discounted-UCB (D-UCB) and Sliding-Window UCB (SW-UCB) to exponentially decay historical data and adapt to CTR regime shifts."*

---

### 3. Execution of the Master Pipeline (1.5 Minutes)
> *"Let's execute our master CLI pipeline: `python main.py`."*

*(Run `python main.py` in the terminal and highlight terminal outputs)*

> *"Notice the pipeline outputs across all 5 benchmark stages:
> - **Stage 1 (UCB1):** Out of 10,000 rounds, the agent successfully routes over 51% of traffic to the winning Ad 4 (15% CTR) with sublinear logarithmic regret.
> - **Stage 2 (Benchmark Arena):** Thompson Sampling and decaying $\varepsilon$-greedy harvest over 1,460 clicks—representing an **+85% click improvement** over static A/B testing (806 clicks).
> - **Stage 3 (LinUCB):** Contextual personalization achieves an astounding 65.6% CTR by serving ads tailored to individual demographics.
> - **Stage 4 (Non-Stationary):** When Ad 4 suffers ad fatigue at round 5,000, Sliding-Window UCB achieves a **13.8% reduction in regret** over standard stationary UCB1.
> - **Stage 5 (Offline Replay):** Unbiased evaluation on 10,000 logged impression records proves LinUCB delivers highest real-world CTR (47.9%)."*

---

### 4. Interactive Academic Dashboard Walkthrough (4 to 5 Minutes)
> *"Now let's launch our interactive academic dashboard: `streamlit run dashboard/app.py`."*

*(Open browser at `http://localhost:8501` and walk through the tabs)*

#### Tab 1: Single-Agent UCB1 Deep Dive
- Show sidebar sliders (simulation rounds, exploration coefficient $c$, custom CTRs).
- Point out KPI cards (Total Clicks, Average CTR, Optimal Ad, Winner Identified).
- Highlight the **Sublinear Cumulative Regret** curve (flattens logarithmically, validating theoretical bounds).
- Highlight the **True CTR vs. Estimated CTR Convergence** bar chart.

#### Tab 2: Multi-Algorithm Benchmark Arena
- Show side-by-side leaderboard ranking UCB1, Thompson Sampling, $\varepsilon$-Greedy, and Random A/B Testing.
- Point to the **Cumulative Regret Comparison** chart:
  - Explain why Random and fixed $\varepsilon$-greedy exhibit **steep linear regret** ($O(T)$).
  - Explain why UCB1 and Thompson Sampling exhibit **flattening logarithmic curves** ($O(\ln T)$).
- Point to the Total Clicks comparison (+85% click lift).

#### Tab 3: Contextual Personalization (LinUCB)
- Demonstrate the **Live User Persona Simulator**:
  - Check "Mobile User" + "Age Under 30" + "High Tech Affinity" $\to$ LinUCB immediately recommends **Gaming App Ad**.
  - Uncheck "Age Under 30" and "Mobile" $\to$ LinUCB immediately recommends **Financial Investment Ad** or **Cloud SaaS Ad**.
- Explain the ridge regression update mechanism without black-box ML libraries.

#### Tab 4: Non-Stationary Dynamics & Ad Fatigue
- Demonstrate the shift point slider (round 5,000) where Ad 4 collapses from 15% to 2% and Ad 2 surges to 18%.
- Show the visual comparison graph proving how stationary UCB1 gets stuck on stale impressions while Discounted-UCB adapts and shifts traffic to the new winner.

#### Tab 5: Unbiased Offline Policy Evaluation
- Explain the Li et al. (WWW 2010) Replay Method.
- Show the accepted impression statistics and effective CTR trajectory curves across 10,000 logged impression events.

#### Tab 6: Academic Viva & Theory Compendium
- Show the built-in oral viva revision handbook covering Hoeffding bounds, Lai-Robbins theorem, Beta conjugate priors, and trade-offs.

---

### 5. Automated Testing & Verification (1 Minute)
> *"To ensure software engineering rigor, we have written 36 comprehensive unit tests across 6 modules covering mathematical bound computations, stochastic sampling, and edge cases. Running `pytest -v`:"*

*(Run `pytest -v` in terminal)*
> *"All 36 unit tests pass in 1.25 seconds with 100% success rate."*

---

### 6. Closing & Q&A Transition
> *"In summary, this project delivers a mathematically verified, multi-tiered reinforcement learning platform for online advertising. We have bridged theoretical optimality proofs with practical personalized serving and offline validation. Thank you, and I welcome any questions."*

---

## ⚡ Quick Benchmark Cheat Sheet (Final Submission)

| Metric | Benchmark Value | Reference Insight |
| :--- | :--- | :--- |
| **Candidate Ads** | 5 Ads (`Ad 1` to `Ad 5`) | True CTRs: $[0.05, 0.10, 0.03, 0.15, 0.08]$ |
| **Static A/B Clicks** | ~806 clicks | Baseline uniform random exploration |
| **UCB1 Clicks** | ~1,126 clicks | +39.7% lift over static testing |
| **Thompson Sampling Clicks**| ~1,464 clicks | +81.6% lift over static testing |
| **Decaying $\varepsilon$-Greedy** | ~1,491 clicks | +85.0% lift over static testing |
| **Contextual LinUCB CTR** | ~65.6% personalized CTR | Over 4x higher than best single static ad |
| **SW-UCB Non-Stationary Lift**| 13.8% Regret Reduction | Rapid recovery from 15% $\to$ 2% ad fatigue |
| **Unit Test Coverage** | 36 / 36 Passing (100%) | Verified in Python 3.11 |
