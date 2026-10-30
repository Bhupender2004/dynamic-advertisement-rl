# 📖 Student & User Guide: Dynamic Advertisement Selection Dashboard

Welcome to the **Dynamic Advertisement Selection Using Upper Confidence Bound Reinforcement Learning** research platform. This concise guide explains how to start, configure, run, and interpret experiments using the interactive web dashboard.

---

## 1. How to Launch the Application

1. Open your terminal in the project root directory:
   ```bash
   cd dynamic-advertisement-rl
   ```
2. Activate your virtual environment:
   * **Windows (PowerShell):** `.\venv\Scripts\Activate.ps1`
   * **macOS / Linux:** `source venv/bin/activate`
3. Launch the dashboard:
   ```bash
   streamlit run dashboard/app.py
   ```
4. Open your browser at **`http://localhost:8501`**.

---

## 2. What Each Input Parameter Means

| Parameter | Location | Meaning | Recommended Default |
| :--- | :--- | :--- | :---: |
| **Number of Rounds ($T$)** | Top Configuration | The total number of sequential visitor impressions served in the closed-loop simulation. | `10,000` |
| **Exploration Coefficient ($c$)** | Top Configuration | Controls the width of the confidence interval derived from Hoeffding's Inequality. Higher values ($c > 3$) force aggressive testing of uncertain ads; lower values ($c < 0.5$) make the agent greedy. | `2.0` |
| **Random Seed** | Top Configuration | Integer seed for the random number generator ensuring reproducible Bernoulli rewards. | `42` |
| **Advertisement CTRs** | "Configure Hidden CTRs" dropdown | The true underlying click probability for each candidate advertisement ($\mathbb{P}(\text{Click} = 1)$). The RL agent **never** sees these values. | `[0.05, 0.10, 0.03, 0.15, 0.08]` |

---

## 3. How to Configure Advertisements

1. Click on the expander labeled **"⚙️ Advertisement Ground Truth CTRs (Environment)"**.
2. Adjust the number input for each advertisement from `0.00` to `1.00`.
3. For example, set **Ad 4 CTR = 0.20** and all other ads to `0.05` to observe how rapidly the agent identifies and routes traffic to Ad 4.

---

## 4. How to Run a Simulation (Configuration → Action → Results)

> [!IMPORTANT]
> Changing an input value **does NOT automatically execute the simulation**. This prevents lag and accidental computation.

1. Adjust your desired parameters in the inputs.
2. Click the black button labeled **`▶ Run Simulation`**.
3. A progress spinner appears while the closed-loop interaction loop executes.
4. Results, metrics scorecards, plots, and the performance matrix will render immediately.
5. **If you change a parameter after running:**
   - Previous results will remain safely visible.
   - An amber notice banner will appear:  
     `⚠️ Configuration changed. Click "Run Simulation" to update results.`
   - Click **`▶ Run Simulation`** again to update results with the new values.

---

## 5. How to Reset the Configuration

1. Click the white button labeled **`↺ Reset`**.
2. All inputs will restore to their initial default benchmark values (`T = 10,000`, `c = 2.0`, `Seed = 42`).
3. Results are cleared back to the clean initial empty state.

---

## 6. How to Understand Each Output Metric

* **Total Reward (Clicks):** Total clicks harvested across all rounds ($\sum r_t$). Higher is better.
* **Average System CTR:** Overall platform conversion rate ($\frac{\text{Total Clicks}}{T}$).
* **Optimal Advertisement:** The ad with the highest true hidden CTR (Ground Truth).
* **Cumulative Regret:** The total expected click loss accumulated by testing sub-optimal ads ($\sum (\mu^* - \mu_{a_t})$). In a learning bandit, this curve flattens over time.

---

## 7. How to Interpret Each Graph

### Graph 1: Cumulative Reward Over Time
* **X-Axis:** Simulation Round ($t$).
* **Y-Axis:** Cumulative Clicks.
* **Interpretation:** A steady upward slope. The steeper the slope, the higher the click yield.

### Graph 2: Cumulative Regret Over Time
* **X-Axis:** Simulation Round ($t$).
* **Y-Axis:** Expected Regret Deficit.
* **Interpretation:** The curve flattens out (sublinear logarithmic profile $O(\ln T)$). A flattening curve is the mathematical proof that the agent has learned the best ad and stopped making mistakes.

### Graph 3: Advertisement Selection Frequency
* **X-Axis:** Candidate Ads (`Ad 1` to `Ad 5`).
* **Y-Axis:** Impressions Allocated.
* **Interpretation:** The optimal ad (green bar) should dominate total impressions ($>50\%$), while inferior ads receive only initial exploratory traffic.

### Graph 4: True CTR vs. Estimated CTR Convergence
* **X-Axis:** Candidate Ads.
* **Y-Axis:** Click-Through Rate.
* **Interpretation:** Compares ground truth (slate bars) against what the UCB agent learned empirically (amber bars). They should closely match, demonstrating accurate statistical estimation without cheating.

---

## 8. Exploring the Other Research Tabs

* **Tab 2 (Algorithm Arena):** Click **`⚔️ Run Benchmark Arena`** to see UCB1, Thompson Sampling, $\varepsilon$-Greedy, and Static Random A/B testing compete side-by-side.
* **Tab 3 (Contextual Personalization):** Toggle user traits (**Mobile**, **Age Under 30**, **Tech Affinity**) to see which ad LinUCB serves to that specific visitor persona.
* **Tab 4 (Non-Stationary Dynamics):** Click **`🔄 Run Non-Stationary Simulation`** to see how Discounted-UCB adapts when the best ad suffers ad fatigue at round 5,000.
* **Tab 5 (Offline Replay Evaluation):** Click **`📊 Run Offline Replay Evaluation`** to test the algorithms on 10,000 historical impression records without live traffic.
* **Tab 6 (Methodology & Viva Guide):** Read quick answers to common questions asked by academic examiners and project reviewers.
