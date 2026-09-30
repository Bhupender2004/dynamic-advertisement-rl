# Academic Viva Preparation: Questions & Answers

**Project Title:** Dynamic Advertisement Selection Using Upper Confidence Bound Reinforcement Learning  
**Core Domain:** Reinforcement Learning / Multi-Armed Bandits / Online Optimization  

---

### Q1: What is Reinforcement Learning?
**Answer:** Reinforcement Learning (RL) is a subfield of machine learning where an autonomous agent learns to make sequential decisions by interacting with an environment. Rather than receiving explicit training labels, the agent receives feedback in the form of scalar rewards (positive or negative) and seeks to learn an optimal policy that maximizes long-term cumulative reward.

---

### Q2: What is a Multi-Armed Bandit (MAB) problem?
**Answer:** A Multi-Armed Bandit is a foundational simplified formulation of Reinforcement Learning where an agent faces $K$ different choices (arms/actions), each with an unknown probability distribution of rewards. The objective is to maximize total accumulated reward over $T$ time steps by balancing between trying unfamiliar arms to gather information and picking the best-known arm.

---

### Q3: What is the difference between a Multi-Armed Bandit and a standard Markov Decision Process (MDP)?
**Answer:**
* **MDP:** Involves full sequential states where actions transition the agent from state $s$ to state $s'$, and current actions influence future states and delayed rewards.
* **MAB:** Is a stateless (or single-state) non-associative RL setting. Actions do not alter the environmental state or future available choices; each round represents an independent trial with immediate reward feedback.

---

### Q4: What is the Exploration vs. Exploitation dilemma?
**Answer:**
* **Exploration:** Choosing arms with high uncertainty to gain new information about their true reward distributions.
* **Exploitation:** Choosing the arm that currently has the highest estimated reward to maximize immediate payoff.
* **Dilemma:** Focusing purely on exploitation can trap the agent on a sub-optimal choice (premature convergence), while focusing purely on exploration wastes opportunities to collect maximum rewards.

---

### Q5: Why is online advertisement selection modeled as a Multi-Armed Bandit problem?
**Answer:** When an ad platform has multiple ads to serve, it does not know their true Click-Through Rates (CTRs) in advance. Static A/B testing wastes impressions on low-converting ads. By framing ad placement as an MAB, the algorithm dynamically balances testing new/untested ads (exploration) with serving the winning ad to the majority of visitors (exploitation).

---

### Q6: What is the Upper Confidence Bound (UCB) algorithm?
**Answer:** UCB is a deterministic bandit algorithm based on the principle of **"Optimism in the Face of Uncertainty"**. For each arm, it constructs a confidence interval around its empirical sample mean and always selects the arm with the highest plausible upper bound.

---

### Q7: Explain the UCB1 formula step-by-step.
**Answer:**
$$\text{UCB}_i(t) = \hat{\mu}_i + c \sqrt{\frac{\ln(t)}{N_i}}$$
* $\hat{\mu}_i$: Empirical sample mean reward ($\frac{\text{Total Clicks on Ad } i}{\text{Selections of Ad } i}$). Represents **exploitation**.
* $c$: Exploration coefficient scaling the confidence level (typically $2.0$ or $\sqrt{2}$).
* $t$: Total rounds elapsed across all ads so far.
* $N_i$: Number of times ad $i$ has been selected.
* $\sqrt{\frac{\ln(t)}{N_i}}$: The uncertainty radius derived from Hoeffding's Inequality. Represents **exploration**.

---

### Q8: Why is the natural logarithm $\ln(t)$ used in the numerator?
**Answer:** $\ln(t)$ grows very slowly as $t$ increases. This slow logarithmic growth ensures that the exploration bonus shrinks rapidly for frequently selected arms (where $N_i$ grows linearly), but never drops to zero, ensuring that arms that were unlucky early on still get re-tested over long horizons.

---

### Q9: What happens when an advertisement has never been selected ($N_i = 0$)?
**Answer:** In the UCB1 formula, if $N_i = 0$, division by zero occurs. Theoretically, its uncertainty radius is $\infty$. Therefore, UCB initializes by pulling every candidate advertisement at least once during the first $K$ rounds, ensuring $N_i \ge 1$ for all arms before applying the formula.

---

### Q10: What is the effect of tuning the exploration coefficient $c$?
**Answer:**
* **High $c$ ($c > 3.0$):** Forces aggressive exploration. The agent spends more rounds testing lower-performing ads before committing to the best ad.
* **Low $c$ ($c < 0.5$):** Leads to greedy behavior. The agent quickly commits to the first ad that gets an early click, risking premature convergence to a sub-optimal ad.

---

### Q11: What is the reward in our simulation?
**Answer:** The reward is binary ($r_t \in \{0, 1\}$), where $1$ indicates a user clicked the displayed ad and $0$ indicates no click. The environment generates this reward using a Bernoulli trial: $r_t \sim \text{Bernoulli}(p = \text{CTR}_i)$.

---

### Q12: What is Click-Through Rate (CTR)?
**Answer:** CTR is the ratio of clicks to impressions:
$$\text{CTR} = \frac{\text{Total Clicks}}{\text{Total Impressions}}$$
In our simulation, each ad has a fixed, hidden true CTR that parameterizes the Bernoulli distribution.

---

### Q13: What is Regret, and how is it defined?
**Answer:** Regret measures the loss incurred by not picking the optimal arm in hindsight.
* **Instantaneous Expected Regret at round $t$:**
  $$\delta_t = \mu^* - \mu_{a_t}$$
  where $\mu^* = \max_i \text{CTR}_i$ and $\mu_{a_t}$ is the true CTR of the chosen ad.
* **Cumulative Expected Regret over $T$ rounds:**
  $$\text{Regret}(T) = \sum_{t=1}^T (\mu^* - \mu_{a_t})$$

---

### Q14: Why does cumulative regret grow logarithmically ($O(\ln T)$) instead of linearly?
**Answer:** A random or non-learning policy makes sub-optimal choices at a constant rate, resulting in linear regret ($O(T)$). UCB selects sub-optimal arms at a rate proportional to $O(\ln T)$. As $T \to \infty$, the average regret per round $\frac{\text{Regret}(T)}{T} \to 0$, proving asymptotic optimality.

---

### Q15: Does the UCB agent have access to the true CTRs?
**Answer:** **No.** The UCB agent has zero knowledge of the underlying true CTRs. It only observes the selected action index and the stochastic binary reward returned by the environment. True CTRs are strictly used by the evaluation harness to calculate theoretical regret.

---

### Q16: How does UCB differ from the $\varepsilon$-Greedy algorithm?
**Answer:**
* **$\varepsilon$-Greedy:** Explores randomly with probability $\varepsilon$ across all arms, treating high-uncertainty arms and obviously poor arms identically.
* **UCB:** Explores **quantifiably** and **optimistically**. It selectively explores arms that either show high estimated potential or have high statistical uncertainty.

---

### Q17: How does UCB differ from Thompson Sampling?
**Answer:**
* **UCB:** Is a **frequentist / deterministic** algorithm using confidence bounds derived from concentration inequalities.
* **Thompson Sampling:** Is a **Bayesian / probabilistic** algorithm that maintains posterior probability distributions (e.g., Beta distributions for Bernoulli rewards) over the arms' CTRs and samples from these posteriors to make decisions.

---

### Q18: What is Hoeffding's Inequality and why is it relevant?
**Answer:** Hoeffding's Inequality is a statistical theorem providing an upper bound on the probability that the empirical sample mean $\hat{\mu}$ deviates from the expected value $\mu$. In UCB1, it guarantees with high probability ($1 - t^{-4}$) that the true CTR lies below $\hat{\mu}_i + \sqrt{\frac{2\ln t}{N_i}}$.

---

### Q19: What is the computational complexity of UCB1 per round?
**Answer:** For $K$ advertisements, evaluating the UCB score requires $O(1)$ operations per ad. Finding the maximum across $K$ ads takes $O(K)$ time. Thus, each recommendation step runs in $O(K)$ time with $O(K)$ space complexity, making it extremely fast and suitable for real-time systems.

---

### Q20: What are the main limitations of this Review 1 implementation?
**Answer:**
1. **Stationary CTRs:** Assumes true click probabilities never change over time (no seasonal drift or ad fatigue).
2. **Context-Free:** Treats all users identically; does not incorporate user demographics, browsing history, or device features.
3. **Simulated Payoffs:** Relies on synthetic Bernoulli trials rather than production bidding auctions or click logs.

---

### Q21: What are Contextual Bandits and how will they be implemented for Review 2?
**Answer:** Contextual Bandits (e.g., **LinUCB**) extend MAB by taking a user/context feature vector $x_{t, a}$ (e.g., location, time of day, device) as input. The expected payoff is modeled as a linear relationship $\mathbb{E}[r \mid x] = x^T \theta_a$, enabling personalized ad recommendations. This forms our primary objective for the remaining 50% of the project.
