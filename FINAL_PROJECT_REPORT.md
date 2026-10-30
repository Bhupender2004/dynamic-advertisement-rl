# DYNAMIC ADVERTISEMENT SELECTION USING UPPER CONFIDENCE BOUND REINFORCEMENT LEARNING

<br><br>

**A PROJECT REPORT**

*Submitted in partial fulfillment of the requirements for the degree of*

**BACHELOR OF TECHNOLOGY**  
*in*  
**COMPUTER SCIENCE AND ENGINEERING**

<br>

**Submitted by:**

**[STUDENT NAME]**  
**(Register Number: [REGISTER NUMBER])**

<br>

**Under the Guidance of:**

**[GUIDE NAME]**  
*[DESIGNATION]*  
*Department of Computer Science and Engineering*

<br><br>

<div align="center">

**DEPARTMENT OF COMPUTER SCIENCE AND ENGINEERING**  
**SRM INSTITUTE OF SCIENCE AND TECHNOLOGY**  
**TIRUCHIRAPPALLI CAMPUS**  
**TAMIL NADU, INDIA**

**Academic Year: [ACADEMIC YEAR]**  
**Semester: [SEMESTER]**

</div>

<div style="page-break-after: always;"></div>

---

# CERTIFICATE

This is to certify that the project report entitled **"DYNAMIC ADVERTISEMENT SELECTION USING UPPER CONFIDENCE BOUND REINFORCEMENT LEARNING"** is the bona fide work carried out by **[STUDENT NAME]** (Register Number: **[REGISTER NUMBER]**) in partial fulfillment of the requirements for the award of the degree of **Bachelor of Technology in Computer Science and Engineering** of **SRM Institute of Science and Technology, Tiruchirappalli Campus**, during the academic year **[ACADEMIC YEAR]**.

<br><br><br>

**[GUIDE NAME]**  
Project Guide  
[DESIGNATION]  
Department of Computer Science and Engineering  
SRM Institute of Science and Technology  
Tiruchirappalli Campus  

<br><br><br>

**[HEAD OF DEPARTMENT]**  
Head of the Department  
Department of Computer Science and Engineering  
SRM Institute of Science and Technology  
Tiruchirappalli Campus  

<br><br>

Submitted for the University Project Viva-Voce Examination held on: **[DATE]**

<br><br>

**Internal Examiner** &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **External Examiner**

<div style="page-break-after: always;"></div>

---

# DECLARATION

I, **[STUDENT NAME]** (Register Number: **[REGISTER NUMBER]**), hereby declare that the project work entitled **"DYNAMIC ADVERTISEMENT SELECTION USING UPPER CONFIDENCE BOUND REINFORCEMENT LEARNING"** submitted to the Department of Computer Science and Engineering, SRM Institute of Science and Technology, Tiruchirappalli Campus, in partial fulfillment of the requirements for the award of the degree of **Bachelor of Technology in Computer Science and Engineering**, is a record of original work done by me under the supervision and guidance of **[GUIDE NAME]**, [DESIGNATION], Department of Computer Science and Engineering.

I further declare that this report has not formed the basis for the award of any Degree, Diploma, Associateship, Fellowship, or any other similar title in any University or Institution. All primary theoretical literature, empirical findings, and algorithms referenced from existing literature have been appropriately cited and acknowledged.

<br><br><br>

**Place:** Tiruchirappalli  
**Date:** [DATE]  

<br><br>

**[STUDENT NAME]**  
(Register Number: [REGISTER NUMBER])

<div style="page-break-after: always;"></div>

---

# ACKNOWLEDGEMENT

First and foremost, I express my sincere gratitude to the Almighty for providing the strength, wisdom, and perseverance required to complete this project successfully.

I express my deepest gratitude to the Management and Leadership of **SRM Institute of Science and Technology, Tiruchirappalli Campus**, for providing outstanding academic infrastructure, computing facilities, and an encouraging research environment.

I extend my heartfelt gratitude to **[HEAD OF DEPARTMENT]**, Head of the Department of Computer Science and Engineering, for continuous academic encouragement and administrative support throughout the tenure of this curriculum.

I take immense pride and honor in expressing my profound gratitude and indebtedness to my project guide, **[GUIDE NAME]**, [DESIGNATION], Department of Computer Science and Engineering, for invaluable guidance, constructive criticism, inspiring discussions, and patient supervision throughout the design, implementation, and documentation of this work.

I also extend my sincere thanks to all faculty members, technical laboratory assistants, and staff of the Department of Computer Science and Engineering for their direct and indirect support during the execution of this project.

Finally, I express my deepest love and gratitude to my family and friends for their unwavering faith, patience, understanding, and moral encouragement throughout my academic journey.

<br><br>

**[STUDENT NAME]**  
(Register Number: [REGISTER NUMBER])

<div style="page-break-after: always;"></div>

---

# ABSTRACT

In modern digital advertising ecosystems, maximizing user engagement measured by the Click-Through Rate (CTR) under statistical uncertainty represents a foundational revenue and resource allocation challenge. Traditional static A/B testing partitions impression volume uniformly across candidate advertisements for a static observation period, incurring substantial opportunity cost by repeatedly exposing visitors to demonstrably inferior creatives. 

This project formulates dynamic advertisement allocation as a Multi-Armed Bandit (MAB) Reinforcement Learning problem, resolving the fundamental exploration-exploitation trade-off through the Upper Confidence Bound (UCB1) algorithm. The proposed system is developed from foundational mathematical principles without black-box reinforcement learning libraries. The system comprises: (1) an online stochastic advertisement environment modeling binary user click responses via Bernoulli distributions parameterized by latent CTRs; (2) a closed-loop UCB1 agent balancing empirical sample means with Hoeffding-derived confidence bounds; (3) a Multi-Algorithm Benchmark Arena comparing UCB1 against Thompson Sampling, fixed $\varepsilon$-Greedy, decaying $\varepsilon$-Greedy, and uniform random static allocation; (4) a Contextual Personalization module implementing Disjoint Linear UCB (LinUCB) over 5-dimensional user context vectors; (5) a Non-Stationary Dynamics engine evaluating Discounted-UCB (D-UCB) and Sliding-Window UCB (SW-UCB) under sudden ad fatigue; and (6) an unbiased offline policy replay evaluator based on the Li et al. (WWW 2010) rejection sampling framework.

Empirical evaluation across $T = 10,000$ simulation rounds reveals that UCB1 rapidly converges toward the optimal creative (True CTR = 15%), directing 5,156 impressions (51.56%) to it, achieving 1,126 clicks (11.26% system CTR) with sublinear logarithmic regret ($O(\ln T)$). In multi-agent benchmarks, adaptive policies harvested up to +85.0% more clicks than uniform static allocation (806 clicks, 8.06% CTR). LinUCB demonstrated a personalized system CTR of 65.63%, while Sliding-Window UCB achieved a 13.8% regret reduction over stationary UCB1 during mid-stream regime shifts. A production-grade Streamlit research dashboard with strict configuration-action separation provides real-time experimentation, interactive persona profiling, and automated verification across 42 automated test cases.

<br>

**Keywords:** Reinforcement Learning, Multi-Armed Bandit, Upper Confidence Bound, UCB1, Click-Through Rate, Exploration-Exploitation Dilemma, Regret Minimization, Contextual Bandits, LinUCB, Non-Stationary Bandits, Offline Policy Evaluation.

<div style="page-break-after: always;"></div>

---

# TABLE OF CONTENTS

- **Title Page**
- **Certificate**
- **Declaration**
- **Acknowledgement**
- **Abstract**
- **Keywords**
- **List of Figures**
- **List of Tables**
- **List of Abbreviations**
- **Chapter 1 — Introduction**
  - 1.1 Background
  - 1.2 Problem Statement
  - 1.3 Motivation
  - 1.4 Objectives
  - 1.5 Scope
  - 1.6 Contributions
- **Chapter 2 — Literature Review**
  - 2.1 Reinforcement Learning
  - 2.2 Multi-Armed Bandit Problems
  - 2.3 Exploration-Exploitation Trade-off
  - 2.4 $\varepsilon$-Greedy Strategies
  - 2.5 Upper Confidence Bound (UCB)
  - 2.6 Thompson Sampling (Bayesian MAB)
  - 2.7 Contextual Bandits
  - 2.8 Non-Stationary Bandits
  - 2.9 Advertisement Recommendation Systems
- **Chapter 3 — System Analysis and Requirements**
  - 3.1 Existing / Traditional Approach
  - 3.2 Proposed Approach
  - 3.3 Functional Requirements
  - 3.4 Non-Functional Requirements
  - 3.5 Hardware Requirements
  - 3.6 Software Requirements
- **Chapter 4 — Theoretical Foundation**
  - 4.1 Reinforcement Learning Primitives
  - 4.2 Mapping RL Concepts to Advertisement Selection
  - 4.3 Multi-Armed Bandit Formulation
  - 4.4 Click-Through Rate (CTR)
  - 4.5 Exploration vs. Exploitation
  - 4.6 Upper Confidence Bound (UCB1) Formula
  - 4.7 UCB Decision and Execution Process
  - 4.8 Theoretical Regret and the Lai-Robbins Lower Bound
- **Chapter 5 — System Design**
  - 5.1 System Architecture
  - 5.2 System Workflow
  - 5.3 Module Architecture
  - 5.4 Data Flow
  - 5.5 User Interface Architecture
- **Chapter 6 — Implementation**
  - 6.1 Development Environment
  - 6.2 Advertisement Environment Module
  - 6.3 UCB1 Agent Implementation
  - 6.4 Closed-Loop Simulation Engine
  - 6.5 Reward and Metric Calculations
  - 6.6 Empirical CTR Estimation
  - 6.7 Regret Tracking
  - 6.8 Multi-Agent Benchmark Arena
  - 6.9 Contextual Personalization (LinUCB)
  - 6.10 Non-Stationary Dynamics Engine
  - 6.11 Unbiased Offline Policy Evaluator
  - 6.12 Interactive Research Dashboard
- **Chapter 7 — Experimental Methodology**
  - 7.1 Experimental Setup
  - 7.2 Experimental Procedure
  - 7.3 Evaluation Metrics
  - 7.4 Baseline Algorithms
  - 7.5 Reproducibility and Determinism
- **Chapter 8 — Results and Analysis**
  - 8.1 Single-Agent UCB1 Performance
  - 8.2 Advertisement Selection Distribution
  - 8.3 CTR Convergence and Estimation Accuracy
  - 8.4 Cumulative Reward Trajectory
  - 8.5 Cumulative Regret Analysis
  - 8.6 Multi-Algorithm Benchmark Comparison
  - 8.7 Contextual LinUCB Results
  - 8.8 Non-Stationary Regime Shift Results
  - 8.9 Offline Policy Replay Evaluation Results
  - 8.10 Overall Discussion and Academic Synthesis
- **Chapter 9 — System Verification and Testing**
  - 9.1 Testing Strategy
  - 9.2 Functional and Algorithmic Test Cases
  - 9.3 UCB Algorithm Unit Verification
  - 9.4 Simulation Boundary and Parameter Stress Testing
  - 9.5 Dashboard and UI State Management Testing
  - 9.6 Defensive Error Handling
- **Chapter 10 — System Limitations**
  - 10.1 Synthetic User Feedback Distributions
  - 10.2 Absence of Live Ad-Exchange Auction Dynamics
  - 10.3 Discrete Batch Representation
  - 10.4 Linear Contextual Feature Mapping
- **Chapter 11 — Future Scope and Enhancements**
  - 11.1 Real-World Logged Ad Datasets
  - 11.2 Deep Neural Contextual Bandits
  - 11.3 Real-Time Streaming and Low-Latency Serving
  - 11.4 Budget-Constrained and Bidding Bandits
- **Chapter 12 — Conclusion**
- **References**
- **Appendix A — Core Implementation Source Code**
- **Appendix B — Detailed Experimental Result Tables**
- **Appendix C — Dashboard User Interface Verification**

<div style="page-break-after: always;"></div>

---

# LIST OF FIGURES

| Figure No. | Title | Description |
|:---|:---|:---|
| **Figure 5.1** | System Architecture of the Dynamic Advertisement Selection Framework | High-level data flow diagram showing the closed-loop interaction between the simulated user environment and the UCB1 Reinforcement Learning agent. |
| **Figure 5.2** | UCB Decision and Execution Workflow | Sequence diagram illustrating sequential initialization, UCB score calculation, ad selection, Bernoulli reward generation, and state parameter updates. |
| **Figure 8.1** | Cumulative Reward Trajectory for UCB1 Agent | Cumulative clicks harvested across $T = 10,000$ rounds, exhibiting near-linear growth once the optimal ad is identified. |
| **Figure 8.2** | Cumulative Regret Curve Across Simulation Horizon | Cumulative theoretical regret over $T = 10,000$ rounds, demonstrating the characteristic logarithmic $O(\ln T)$ flattening curve. |
| **Figure 8.3** | Advertisement Selection Frequency Distribution | Bar chart displaying total allocation counts per ad arm; optimal Ad 4 received 5,156 impressions (51.56%). |
| **Figure 8.4** | Empirical Estimated CTR vs. Ground Truth CTR Convergence | Convergence comparison showing empirical CTR estimates asymptotically approaching true CTR values with minimal absolute estimation error. |
| **Figure 8.5** | Combined 4-Panel Single-Agent Performance Overview | Unified publication panel showing cumulative reward, cumulative regret, selection distribution, and CTR estimation errors side by side. |
| **Figure 8.6** | Multi-Agent Comparative Cumulative Reward Trajectories | Benchmark trajectories of UCB1, Thompson Sampling, $\varepsilon$-Greedy (decaying), $\varepsilon$-Greedy ($\varepsilon=0.1$), and Random policies over 10,000 rounds. |
| **Figure 8.7** | Multi-Agent Comparative Cumulative Regret Trajectories | Logarithmic vs. linear regret growth comparison highlighting the substantial opportunity cost of uniform random exploration. |
| **Figure 8.8** | Multi-Agent Benchmark Summary Panel | 4-panel comparative analysis showing cumulative reward curves, regret profiles, final system CTRs, and relative click-lift metrics across algorithms. |
| **Figure 8.9** | Contextual Disjoint LinUCB Personalization Trajectory | Performance metrics of LinUCB conditioning ad allocation on 5-dimensional user context vectors, demonstrating personalized click lift. |
| **Figure 8.10** | Non-Stationary Shift Adaptation and Regret Comparison | Trajectory plot illustrating abrupt regime shift at $t = 5,000$, showing rapid recovery of Sliding-Window UCB vs. sluggish stationary UCB1. |
| **Figure C.1** | Dashboard Initial State (Empty Placeholder) | Academic interface before simulation execution, illustrating structured configuration panels and explicit user control. |
| **Figure C.2** | Single-Agent UCB1 Executed Simulation Results | Executed dashboard displaying summary KPI cards, empirical CTR matrix table, and four publication-quality analytical plots. |
| **Figure C.3** | Multi-Algorithm Benchmark Arena Tab | Comparative tournament view featuring the algorithm leaderboard, cumulative reward trajectories, and regret comparison graphs. |
| **Figure C.4** | Contextual Personalization & Interactive Persona Simulator | Mathematical user feature vector $\mathbf{x} \in \mathbb{R}^5$ selector, real-time LinUCB inference preview, and simulated CTR performance curves. |
| **Figure C.5** | Non-Stationary Dynamics & Ad Fatigue Tab | Mid-stream CTR changepoint simulation comparing stationary UCB1, Discounted-UCB, and Sliding-Window UCB. |
| **Figure C.6** | Unbiased Offline Policy Evaluation Tab | Li et al. (WWW 2010) counterfactual replay evaluation table across 10,000 historical ad impression records. |
| **Figure C.7** | Methodology & Academic Viva Preparation Guide | Comprehensive academic reference section featuring algorithmic proofs, theoretical foundations, and 30 viva questions. |
| **Figure C.8** | Responsive Mobile Viewport Layout Verification | Viewport audit at 375px mobile resolution, verifying column stacking, responsive table layout, and touch-friendly controls. |

<div style="page-break-after: always;"></div>

---

# LIST OF TABLES

| Table No. | Title | Description |
|:---|:---|:---|
| **Table 3.1** | Functional Requirements of the Dynamic Advertisement Selection System | Complete breakdown of functional requirements covering simulation, algorithm execution, metric calculation, and dashboard controls. |
| **Table 3.2** | Software Environment and Technology Stack Specifications | List of core programming languages, runtime environments, scientific computing libraries, and web visualization frameworks. |
| **Table 4.1** | Mapping Reinforcement Learning Primitives to Advertisement Selection | Conceptual equivalence table connecting RL formalisms (Agent, Environment, Action, Reward, Policy) to ad serving components. |
| **Table 7.1** | Experimental Simulation Configuration Parameters | Ground-truth CTR values, total round horizon, exploration coefficient, and random seed parameters for baseline experiments. |
| **Table 8.1** | Single-Agent UCB1 Simulation Summary ($T = 10,000$ Rounds) | Empirical CTR estimates, absolute error, selection counts, selection percentages, and click counts per advertisement arm. |
| **Table 8.2** | Multi-Algorithm Benchmark Arena Leaderboard ($T = 10,000$ Rounds) | Comparative performance ranking of UCB1, Thompson Sampling, $\varepsilon$-Greedy variants, and Random policies. |
| **Table 8.3** | Contextual LinUCB Performance Metrics | Summary of harvested clicks, effective personalized CTR, and cumulative contextual regret under 5-dimensional user features. |
| **Table 8.4** | Non-Stationary Regime Shift Performance Metrics | Regret comparison between stationary UCB1, Discounted-UCB, and Sliding-Window UCB under an abrupt mid-stream CTR shift. |
| **Table 8.5** | Unbiased Offline Policy Replay Evaluation on Historical Logs | Replay evaluation metrics (Accepted impressions, clicks, effective CTR) evaluated across 10,000 logged impressions. |
| **Table 9.1** | Automated System Verification and Test Suite Results | 100% pass verification across unit, mathematical, edge-case, and UI interaction test cases (42 / 42 passed). |
| **Table B.1** | Multi-Algorithm Arm Allocation Breakdown Across Arms | Detailed selection percentage matrix per advertisement arm across all five benchmarked bandit policies. |
| **Table B.2** | Offline Policy Replay Sampling Efficiency | Replay acceptance rates and sample-efficiency analysis across UCB1, Thompson Sampling, and LinUCB. |

<div style="page-break-after: always;"></div>

---

# LIST OF ABBREVIATIONS

| Abbreviation | Expanded Form |
|:---|:---|
| **RL** | Reinforcement Learning |
| **MAB** | Multi-Armed Bandit |
| **UCB** | Upper Confidence Bound |
| **UCB1** | Upper Confidence Bound 1 Algorithm (Auer et al., 2002) |
| **CTR** | Click-Through Rate |
| **MDP** | Markov Decision Process |
| **TS** | Thompson Sampling |
| **LinUCB** | Linear Upper Confidence Bound (Contextual Bandit) |
| **D-UCB** | Discounted Upper Confidence Bound |
| **SW-UCB** | Sliding-Window Upper Confidence Bound |
| **CDF** | Cumulative Distribution Function |
| **PDF** | Probability Density Function |
| **KL** | Kullback-Leibler (Divergence) |
| **FIFO** | First-In, First-Out |
| **UI** | User Interface |
| **UX** | User Experience |
| **KPI** | Key Performance Indicator |
| **CLI** | Command-Line Interface |
| **API** | Application Programming Interface |
| **CI/CD** | Continuous Integration / Continuous Deployment |

<div style="page-break-after: always;"></div>

---

# CHAPTER 1 — INTRODUCTION

## 1.1 Background
Digital display advertising, sponsored e-commerce listings, and content recommendation feeds represent foundational pillars of contemporary Internet economics. In automated ad delivery platforms, publishers receive sequential requests for content exposures, termed impressions. When a visitor arrives at a web property, the ad serving platform must instantaneously choose one or more advertisements to display from a pool of candidate creatives.

The fundamental metric of audience engagement in digital marketing is the Click-Through Rate (CTR), defined as the empirical ratio of user clicks to total served impressions. Maximizing cumulative clicks over a business campaign directly drives advertiser return on investment (ROI) and publisher monetization. However, ad networks operate under severe informational asymmetry: when a new advertisement creative enters the inventory, its true click probability is completely unknown. 

Historically, platforms relied on static A/B testing (or multivariate testing). In a traditional A/B split, incoming traffic is partitioned equally across all candidate ad variations for a predetermined test horizon. Only after statistical significance tests conclude does the publisher identify the winning variation and allocate subsequent traffic to it. While statistically straightforward, this static paradigm incurs severe economic penalties: during the prolonged test window, thousands of prospective users are continuously exposed to suboptimal, low-converting advertisements, generating enormous cumulative opportunity loss. Modern digital systems require adaptive algorithms that continuously learn from live user feedback and dynamically route traffic to superior alternatives without requiring manual intervention.

## 1.2 Problem Statement
Let $\mathcal{A} = \{1, 2, \dots, K\}$ represent a discrete set of $K$ candidate advertisements available for placement. Each advertisement $i \in \mathcal{A}$ is governed by an unknown, stationary ground-truth click probability $\mu_i \in [0, 1]$. At each discrete time step $t \in \{1, 2, \dots, T\}$, the system must select an advertisement $a_t \in \mathcal{A}$ to display to an arriving user. The user generates a stochastic binary engagement response $r_t \in \{0, 1\}$, where $r_t = 1$ denotes a click and $r_t = 0$ denotes an impression without interaction, sampled from a Bernoulli distribution parameterized by the true rate:
$$r_t \sim \text{Bernoulli}(\mu_{a_t})$$

The core optimization problem is formulated as follows:
> *Given a finite inventory of $K$ candidate advertisements with unknown performance characteristics, design an autonomous decision policy $\pi$ that sequentially selects advertisements $a_t$ over an impression horizon $T$ so as to maximize the cumulative harvested rewards $\sum_{t=1}^T r_t$, or equivalently, minimize the cumulative expected opportunity loss (regret) incurred by not displaying the optimal creative at every step.*

## 1.3 Motivation
This dynamic decision-making challenge cannot be effectively solved by standard supervised machine learning because offline labeled training data is unavailable for newly introduced advertisements. Furthermore, the feedback received is purely partial (bandit feedback): the system observes user response *only* for the specific advertisement displayed; no counterfactual information is obtained regarding how the user would have reacted to the unselected advertisements.

Consequently, the problem maps directly to the classical **Multi-Armed Bandit (MAB)** formulation of Reinforcement Learning (RL). The agent is confronted with the fundamental **Exploration-Exploitation Dilemma**:
1. **Exploitation:** Directing impressions to the advertisement currently exhibiting the highest observed empirical click-through rate, thereby securing immediate reward.
2. **Exploration:** Directing impressions to less-frequently displayed advertisements whose true click probabilities remain statistically uncertain, thereby acquiring information to prevent premature convergence to a suboptimal creative.

Heuristic strategies such as uniform random exploration or fixed $\varepsilon$-greedy exploration incur linear cumulative regret $O(T)$, continuously wasting impressions indefinitely. In contrast, the **Upper Confidence Bound (UCB1)** algorithm, rooted in the mathematical principle of *Optimism in the Face of Uncertainty*, constructs confidence bounds using concentration inequalities (Hoeffding's Inequality). UCB1 achieves provably optimal logarithmic regret $O(\ln T)$, guaranteeing that cumulative regret grows at the theoretically slowest possible rate as demonstrated by Lai and Robbins (1985).

## 1.4 Objectives
The primary technical and academic objectives of this project are:
1. **Stochastic Environment Modeling:** Design and implement a modular, reproducible Multi-Armed Bandit simulation environment capable of generating stochastic binary feedback for an arbitrary number of candidate advertisements with configurable ground-truth CTRs.
2. **Algorithmic UCB1 Implementation:** Develop the classical UCB1 decision algorithm from foundational mathematical principles without third-party reinforcement learning libraries, incorporating proper cold-start arm initialization and Hoeffding confidence bound scaling.
3. **Comprehensive Performance Evaluation:** Quantify algorithmic performance over extensive simulation horizons ($T = 10,000$ rounds) by tracking cumulative reward, instantaneous regret, cumulative theoretical regret, selection frequencies, and empirical CTR convergence rates.
4. **Multi-Agent Benchmarking:** Construct an Algorithm Benchmark Arena to compare UCB1 against four foundational baseline policies: Thompson Sampling (Bayesian Beta-Bernoulli conjugate priors), fixed $\varepsilon$-Greedy ($\varepsilon = 0.1$), decaying $\varepsilon$-Greedy ($\varepsilon_t \propto K/t$), and Uniform Random static allocation.
5. **Contextual Feature Extension:** Formulate and implement Disjoint Linear Upper Confidence Bound (LinUCB) to demonstrate personalization by conditioning advertisement selection on continuous multi-dimensional user context vectors $x_t \in \mathbb{R}^5$ (e.g., device type, demographic age, temporal browsing window, tech affinity).
6. **Non-Stationary Adaptation:** Implement adaptive bandit variants (Discounted-UCB and Sliding-Window UCB) and quantify their recovery performance under abrupt mid-stream CTR shifts simulating creative fatigue.
7. **Unbiased Offline Policy Evaluation:** Implement the Li et al. (WWW 2010) offline replay evaluation methodology to validate policy learning on logged historical impression data without risking live production traffic.
8. **Interactive Research Dashboard:** Engineer a high-usability academic research dashboard in Streamlit adhering to a strict **Configuration $\to$ Action $\to$ Result** interaction model, publication-quality visualizations, and zero automated auto-run side effects.

## 1.5 Scope
### Included Within Project Scope:
- Formulation of advertisement placement as a Bernoulli Multi-Armed Bandit problem.
- Complete object-oriented Python architecture implementing base agent interfaces, simulation harnesses, metric calculation utilities, and visualization modules.
- Implementation of five distinct decision algorithms: UCB1, Thompson Sampling, $\varepsilon$-Greedy (fixed and decaying), LinUCB, and Non-Stationary UCB variants (D-UCB, SW-UCB).
- Offline policy evaluation engine using counterfactual replay rejection sampling.
- Interactive academic dashboard with real-time persona simulation and mathematical vector inspection.
- Automated testing suite comprising 42 unit, simulation, algorithmic, and UI interaction test cases with 100% pass verification.

### Excluded From Project Scope:
- Live integration with real-time advertising exchanges (e.g., OpenRTB, Google Ad Manager APIs).
- Continuous real-time financial bidding and budget-pacing mechanisms (Second-Price Auctions, Vickrey-Clarke-Groves mechanisms).
- High-volume distributed streaming backends (e.g., Apache Kafka, Apache Flink).

## 1.6 Contributions
This work makes the following concrete contributions:
- **Foundational, Dependency-Free Implementation:** All algorithms (UCB1, Thompson Sampling, LinUCB, D-UCB, SW-UCB) are built from pure mathematical specifications using only NumPy and Pandas, ensuring complete transparency and educational value.
- **Empirical Validation of Regret Bounds:** Provides empirical verification of theoretical logarithmic regret bounds ($O(\ln T)$) versus linear regret ($O(T)$) under controlled, reproducible conditions.
- **Unified Academic Framework:** Integrates stationary MAB, contextual bandits, non-stationary dynamics, and offline evaluation into a single, cohesive, publication-grade academic laboratory platform.

<div style="page-break-after: always;"></div>

---

# CHAPTER 2 — LITERATURE REVIEW

## 2.1 Reinforcement Learning
Reinforcement Learning (RL) is a paradigm of machine learning concerned with how an autonomous agent ought to take sequential actions in an environment to maximize a cumulative numerical reward signal [3]. Unlike supervised learning, where an agent learns from a supervisor providing explicit input-target pairs, an RL agent must discover which actions yield the highest reward by trying them [3]. The classical formulation of Reinforcement Learning relies on Markov Decision Processes (MDPs), defined by a 5-tuple $(S, A, P, R, \gamma)$ representing states, actions, transition probabilities, rewards, and a temporal discount factor.

## 2.2 Multi-Armed Bandit Problems
The Multi-Armed Bandit (MAB) problem, originally introduced in mathematical statistics by Robbins (1952), represents the foundational simplification of Reinforcement Learning [3], [10]. In an MAB, the environment is stateless (or consists of a single state), and actions do not alter the future distribution of rewards. Formally, an agent faces $K$ independent arms (actions). When arm $i$ is pulled at round $t$, the agent receives a scalar reward sampled from an unknown stationary probability distribution $P_i$ with expectation $\mu_i = \mathbb{E}[R_i]$. The objective is to maximize cumulative payoff over a discrete horizon $T \in \mathbb{N}$, or equivalently, to minimize cumulative regret against an oracle policy that always selects the arm with maximal expected reward $\mu^* = \max_i \mu_i$ [9], [10].

## 2.3 Exploration-Exploitation Trade-off
The tension between gathering information and capitalizing on existing knowledge is known as the Exploration-Exploitation dilemma. Pure exploration minimizes epistemic uncertainty regarding arm reward distributions but suffers severe regret by repeatedly sampling suboptimal arms. Pure exploitation maximizes immediate payoff based on historical sample means but risks permanent entrapment in a suboptimal action if early stochastic samples of the truly optimal arm were uncharacteristically poor. Balancing this trade-off with mathematical guarantees is the central challenge of bandit theory [9].

## 2.4 $\varepsilon$-Greedy Strategies
The $\varepsilon$-Greedy algorithm is one of the oldest heuristics for addressing the exploration-exploitation dilemma [3]. At each round, the agent selects the arm with the highest empirical mean with probability $1 - \varepsilon$, and selects an arm uniformly at random with probability $\varepsilon$. While intuitive, fixed $\varepsilon$-greedy suffers a fundamental flaw: because exploration continues at a constant rate $\varepsilon > 0$, the agent continues to sample suboptimal arms even as $t \to \infty$, resulting in linear cumulative regret $O(\varepsilon T)$ [9]. Decaying $\varepsilon$-greedy policies introduce a time-dependent exploration schedule, such as $\varepsilon_t = \min(1.0, cK/t)$, which reduces exploration over time and can theoretically achieve logarithmic regret if the sub-optimality gaps $\Delta_i$ are known *a priori*; however, tuning such schedules in practice is notoriously brittle [3], [9].

## 2.5 Upper Confidence Bound (UCB)
To overcome the limitations of random exploration, Lai and Robbins (1985) proved that the asymptotic cumulative regret of any uniformly efficient bandit policy is lower-bounded by:
$$\liminf_{T \to \infty} \frac{\text{Regret}(T)}{\ln T} \ge \sum_{i: \mu_i < \mu^*} \frac{\Delta_i}{D_{\text{KL}}(\mu_i \parallel \mu^*)}$$
where $\Delta_i = \mu^* - \mu_i$ is the sub-optimality gap and $D_{\text{KL}}$ is the Kullback-Leibler divergence between reward distributions [2].

In a seminal paper, Auer, Cesa-Bianchi, and Fischer (2002) introduced **UCB1**, a deterministic algorithm embodying the principle of *Optimism in the Face of Uncertainty* [1]. UCB1 constructs an uncertainty radius around the sample mean of each arm using Hoeffding's Inequality [12]. By always selecting the arm with the maximum upper bound, UCB1 guarantees finite-time logarithmic regret $O(\ln T)$ uniformly over all time horizons without requiring prior knowledge of the reward gaps $\Delta_i$ [1].

## 2.6 Thompson Sampling (Bayesian MAB)
Originally formulated by William R. Thompson in 1933 for clinical trial allocation, Thompson Sampling (also known as Posterior Sampling or Probability Matching) provides a Bayesian approach to bandit optimization [4]. For Bernoulli rewards, conjugate Beta priors $\text{Beta}(\alpha_i, \beta_i)$ are assigned to each arm. At each round, a sample is drawn from each arm's posterior distribution, and the arm yielding the largest sampled value is selected. Agrawal and Goyal (2012) proved that Thompson Sampling achieves the theoretical Lai-Robbins logarithmic lower bound [5]. Extensive empirical evaluations by Chapelle and Li (2011) demonstrated that Thompson Sampling frequently matches or outperforms UCB algorithms in practical internet recommendation tasks [8].

## 2.7 Contextual Bandits
Standard MAB formulations assume that all impressions are identical and drawn from the same distribution. In reality, user click propensities vary significantly based on contextual features (e.g., user demographics, device types, historical interests, geographic region). Contextual bandits bridge the gap between stateless MABs and full MDPs by conditioning arm rewards on an observable feature vector $x_t \in \mathbb{R}^d$ [6], [10]. 

Li et al. (WWW 2010) introduced **LinUCB** for personalized news headline recommendations at Yahoo! [6]. LinUCB models the expected payoff of each arm as a linear function of the user context vector, using ridge regression to estimate weight parameters and an analytical confidence ellipsoid to compute upper confidence bounds in real time [6].

## 2.8 Non-Stationary Bandits
In dynamic web environments, advertisement performance is rarely stationary. Over time, creatives experience "ad fatigue" (diminishing CTRs as user novelty wanes), audience demographic shifts occur, and promotional campaigns alter user interests. Standard UCB1 treats observations from round 1 with the same statistical weight as observations from round $10,000$, resulting in slow adaptation to environmental regime shifts [7]. 

To handle time-varying reward distributions, Garivier and Moulines (2011) introduced **Discounted-UCB (D-UCB)** and **Sliding-Window UCB (SW-UCB)** [7]. D-UCB exponentially discounts older observations using a geometric forgetting factor $\gamma \in (0, 1)$, while SW-UCB computes empirical means and confidence bounds strictly over a sliding historical window of size $\tau$ [7].

## 2.9 Advertisement Recommendation Systems
Bandit algorithms have become central to algorithmic online advertising at leading platforms including Google, Meta, Amazon, and Microsoft [6], [8]. Compared to static split-testing, bandit architectures automatically divert traffic away from poorly performing creatives toward high-converting creatives, reducing the cost of exploration while providing statistical confidence in campaign optimization [6].

<div style="page-break-after: always;"></div>

---

# CHAPTER 3 — SYSTEM ANALYSIS AND REQUIREMENTS

## 3.1 Existing / Traditional Approach
Traditional digital advertising platforms and marketing teams rely on static A/B testing:
1. **Equal Partitioning:** Traffic is split evenly ($1/K$ per creative) across all candidate advertisements.
2. **Fixed Duration:** The test is executed for a fixed horizon (e.g., 2 to 4 weeks or a predetermined sample size such as 50,000 impressions per variant).
3. **Hypothesis Testing:** At the conclusion of the test, hypothesis tests (such as two-sample t-tests or chi-squared tests) are calculated to establish whether one variant statistically outperforms the control at a significance level $\alpha = 0.05$.
4. **Manual Switchover:** If a statistically significant winner is identified, engineers or campaign managers manually reconfigure ad delivery to route 100% of traffic to that variant.

### Deficiencies of the Traditional Approach:
- **Excessive Opportunity Loss:** If an ad creative has a 2% CTR while another has a 15% CTR, static A/B testing routes tens of thousands of users to the 2% creative throughout the entire testing window, accumulating linear regret $O(T)$.
- **Inability to Adapt to Changing Trends:** If creative fatigue occurs after a winning ad is chosen, static systems cannot automatically re-explore alternatives without initiating a completely new A/B test.
- **Lack of Personalization:** A single global winner is selected, disregarding demographic or contextual differences across audience segments.

## 3.2 Proposed Approach
The proposed system implements an autonomous, closed-loop Reinforcement Learning framework based on Upper Confidence Bound multi-armed bandits:
1. **Dynamic Real-Time Allocation:** At every individual impression $t$, the system computes confidence bounds combining empirical sample means with uncertainty radiuses, automatically adjusting allocation percentages.
2. **Optimal Trade-Off:** Arms that perform well receive immediate exploitation traffic, while arms with high statistical uncertainty receive targeted exploration impressions.
3. **Logarithmic Regret:** As proven by Auer et al. (2002), cumulative regret is mathematically bounded by $O(\ln T)$, eliminating the wasteful linear loss of static testing.
4. **Contextual and Non-Stationary Extensions:** The platform supports user-level feature conditioning via LinUCB and adaptive windowing for time-varying CTR shifts.

## 3.3 Functional Requirements
The system functional requirements are formally detailed in Table 3.1.

**Table 3.1: Functional Requirements of the Dynamic Advertisement Selection System**

| Requirement ID | Requirement Name | Description |
|:---|:---|:---|
| **FR-01** | Stochastic Environment Initialization | The environment must initialize $K \ge 2$ advertisement arms with configurable, hidden true CTRs in $[0.0, 1.0]$ and support reproducible seeding. |
| **FR-02** | Bernoulli Feedback Simulation | The environment must sample binary click rewards ($r_t \in \{0, 1\}$) from Bernoulli distributions parameterized by the selected arm's true CTR. |
| **FR-03** | Cold-Start Initialization | The UCB1 agent must select every available arm at least once during initial rounds ($t \le K$) to avoid zero-division in confidence bounds. |
| **FR-04** | UCB Score Computation | For each round $t > K$, the agent must compute $\text{UCB}_i(t) = \hat{\mu}_i + c \sqrt{\ln(t)/N_i}$ and select the arm maximizing the score. |
| **FR-05** | Online State Updates | The agent must update selection counts $N_i$, total rewards $R_i$, and empirical CTRs $\hat{\mu}_i$ in $O(1)$ time upon observing feedback. |
| **FR-06** | Regret & Trajectory Tracking | The simulation harness must log round-by-round time-series trajectories of rewards, instantaneous regret, and cumulative regret. |
| **FR-07** | Multi-Agent Benchmarking | The system must benchmark UCB1 against Thompson Sampling, fixed $\varepsilon$-Greedy, decaying $\varepsilon$-Greedy, and Uniform Random under identical seeds. |
| **FR-08** | Contextual LinUCB Simulation | The system must simulate 5-dimensional user context vectors $x_t \in \mathbb{R}^5$ and execute Disjoint LinUCB with ridge regression updates. |
| **FR-09** | Non-Stationary Drift Adaptation | The system must simulate abrupt CTR regime shifts and evaluate Sliding-Window UCB and Discounted-UCB recovery rates. |
| **FR-10** | Offline Policy Replay | The system must evaluate target policies against logged historical ad click datasets using the Li et al. (WWW 2010) rejection sampling replay method. |
| **FR-11** | Explicit Execution Control | The dashboard must separate Configuration State from Executed State; parameter adjustments must NEVER trigger automatic computation without clicking an action button. |
| **FR-12** | State Mutation Indicators | If input parameters are modified after a simulation run, the dashboard must preserve current results and display an explicit update notice. |

## 3.4 Non-Functional Requirements
- **Performance & Scalability:** The core simulation loop must execute $T = 10,000$ rounds in under 0.5 seconds on standard commodity hardware using vectorized NumPy operations.
- **Reproducibility:** Simulations must yield bitwise-deterministic numerical results when initialized with identical random seeds.
- **Maintainability & Modularity:** Adhere to object-oriented software engineering principles with clear separation between agents, environments, runners, and presentation layers.
- **Usability & Aesthetic Restraint:** The user interface must present a clean, academic laboratory aesthetic avoiding gratuitous animations, neon styling, or unstructured layouts.

## 3.5 Hardware Requirements
The system is lightweight and requires no specialized computing hardware (no GPU acceleration or distributed clusters required):
- **Processor:** Intel Core i3 / AMD Ryzen 3 or higher (Dual-Core, 2.0 GHz minimum).
- **RAM:** Minimum 4 GB RAM (8 GB recommended for interactive browser visualization).
- **Storage:** Minimum 500 MB free disk space for simulation datasets, test trajectories, and generated figures.
- **Display:** Minimum 1280 $\times$ 720 screen resolution (optimized for 1920 $\times$ 1080 desktop and responsive down to 375px mobile viewports).

## 3.6 Software Requirements
The software environment specifications are detailed in Table 3.2.

**Table 3.2: Software Environment and Technology Stack Specifications**

| Component | Technology / Library | Version Specified | Purpose |
|:---|:---|:---|:---|
| **Programming Language** | Python | 3.10+ (Verified on 3.11.9) | Core computational language |
| **Numerical Computation** | NumPy | $\ge 1.24.0$ | Vectorized array operations and stochastic sampling |
| **Data Manipulation** | Pandas | $\ge 2.0.0$ | Trajectory tracking, tabular summaries, and CSV I/O |
| **Visualization** | Matplotlib | $\ge 3.7.0$ | Publication-grade static time-series plots |
| **Web Dashboard** | Streamlit | $\ge 1.30.0$ (Verified 1.59+) | Interactive research user interface |
| **Testing Framework** | Pytest | $\ge 7.4.0$ | Automated test execution and assertions |
| **UI Testing** | Streamlit AppTest / Playwright | $\ge 1.40.0$ | Headless UI state and visual QA verification |
| **Operating System** | Platform Independent | Windows 10/11, macOS, Linux | Universal cross-platform compatibility |

<div style="page-break-after: always;"></div>

---

# CHAPTER 4 — THEORETICAL FOUNDATION

## 4.1 Reinforcement Learning Primitives
In standard Reinforcement Learning, the interaction between an autonomous decision-making entity and its surrounding environment is formalized through five core primitives:
1. **Agent:** The learner or decision-maker responsible for selecting actions.
2. **Environment:** The external system with which the agent interacts, responding to actions with state transitions and numerical reward signals.
3. **State ($s \in \mathcal{S}$):** A representation of the current environmental configuration.
4. **Action ($a \in \mathcal{A}$):** A choice available to the agent that influences the environment.
5. **Reward ($r \in \mathbb{R}$):** A scalar feedback signal indicating the immediate quality of an action.
6. **Policy ($\pi$):** The mapping from environmental perceptions to actions, defining the agent's behavior.

## 4.2 Mapping RL Concepts to Advertisement Selection
The mapping between abstract Reinforcement Learning primitives and the physical domain of online advertisement selection is presented in Table 4.1.

**Table 4.1: Mapping Reinforcement Learning Primitives to Advertisement Selection**

| RL Formal Concept | Project Implementation Equivalent | Domain Meaning |
|:---|:---|:---|
| **Agent** | `UCB1Agent`, `LinUCBAgent` | Autonomous ad serving engine choosing which creative to display. |
| **Environment** | `AdvertisementEnvironment` | The audience / digital publisher platform exhibiting stochastic click responses. |
| **Action ($a_t$)** | Ad Selection index $i \in \{0, \dots, K-1\}$ | Displaying a specific advertisement banner or creative to an arriving visitor. |
| **Reward ($r_t$)** | Binary Click Signal $r_t \in \{0, 1\}$ | User engagement: 1 if the visitor clicked the ad, 0 if no click occurred. |
| **Policy ($\pi$)** | UCB decision rule: $\arg\max_i \text{UCB}_i(t)$ | Strategy balancing empirical click rates with confidence bound bonuses. |
| **State ($s_t$)** | Single-state (or Context Vector $x_t$) | In standard MAB: stateless; in Contextual MAB: visitor demographic vector. |
| **Value ($Q(a)$)** | Empirical CTR $\hat{\mu}_i = R_i / N_i$ | Sample average click probability estimated from historical exposures. |

## 4.3 Multi-Armed Bandit Formulation
The Multi-Armed Bandit is a specialized class of stateless MDPs. Formally, let $K$ denote the number of candidate advertisements. Each advertisement $i \in \{1, \dots, K\}$ possesses an unknown true reward distribution $P_i$ with expectation:
$$\mu_i = \mathbb{E}_{r \sim P_i}[r]$$
In online advertising, rewards are binary ($r \in \{0, 1\}$), meaning each arm's payoff follows a Bernoulli distribution with parameter $\mu_i$:
$$P_i(r = 1) = \mu_i, \quad P_i(r = 0) = 1 - \mu_i$$

At each discrete round $t = 1, 2, \dots, T$:
1. The agent selects arm $a_t \in \{1, \dots, K\}$ according to policy $\pi_t$.
2. The environment generates reward $r_t \sim \text{Bernoulli}(\mu_{a_t})$.
3. The agent observes $r_t$ and updates internal counts and statistics.
4. Crucially, rewards for all unselected arms $j \neq a_t$ remain completely unobserved.

## 4.4 Click-Through Rate (CTR)
The Click-Through Rate represents the empirical probability that an ad exposure produces a user click. For advertisement $i$ after $N_i(t)$ total impressions, its empirical CTR $\hat{\mu}_i(t)$ is computed as:
$$\hat{\mu}_i(t) = \frac{R_i(t)}{N_i(t)} = \frac{\sum_{s=1}^t r_s \cdot \mathbb{I}(a_s = i)}{\sum_{s=1}^t \mathbb{I}(a_s = i)}$$
where $\mathbb{I}(\cdot)$ is the indicator function. By the Strong Law of Large Numbers, as the sample size $N_i(t) \to \infty$, the sample average $\hat{\mu}_i(t)$ converges almost surely to the true click probability $\mu_i$.

## 4.5 Exploration vs. Exploitation
- **Exploitation:** The greedy action selects $a_t^* = \arg\max_i \hat{\mu}_i(t)$. This choice maximizes short-term expected payoff based on current knowledge. However, if early stochastic samples for an inferior ad happen to produce clicks while early samples for the best ad yield zeros, pure exploitation will permanently lock onto the inferior ad.
- **Exploration:** Deliberately sampling arms with low $N_i(t)$ refines the accuracy of $\hat{\mu}_i(t)$. However, allocating too many impressions to exploration depresses total clicks by under-serving the true winner.
- The fundamental objective of UCB is to explore only as much as mathematically necessary to identify the optimal arm with statistical confidence.

## 4.6 Upper Confidence Bound (UCB1) Formula
The UCB1 algorithm (Auer et al., 2002) resolves the exploration-exploitation trade-off using **Hoeffding's Inequality** [1], [12].

### Mathematical Derivation:
Let $X_1, X_2, \dots, X_{N_i}$ be independent and identically distributed random variables bounded in $[0, 1]$ with common mean $\mu_i$. Hoeffding's Inequality states that for any $\epsilon > 0$:
$$\mathbb{P}\left(\hat{\mu}_i - \mu_i \ge \epsilon\right) \le \exp\left(-2 N_i \epsilon^2\right)$$
and symmetrically:
$$\mathbb{P}\left(\mu_i - \hat{\mu}_i \ge \epsilon\right) \le \exp\left(-2 N_i \epsilon^2\right)$$

To construct an upper confidence bound that holds with high probability, we set the right-hand failure probability to a small, decaying threshold $\delta_t = t^{-4}$:
$$\exp\left(-2 N_i \epsilon^2\right) = t^{-4}$$
Solving for the uncertainty radius $\epsilon$:
$$-2 N_i \epsilon^2 = -4 \ln(t) \implies \epsilon^2 = \frac{4 \ln(t)}{2 N_i} = \frac{2 \ln(t)}{N_i} \implies \epsilon = \sqrt{\frac{2 \ln(t)}{N_i}}$$

Generalizing with an exploration scaling parameter $c > 0$, the Upper Confidence Bound score for advertisement $i$ at round $t$ is defined as:
$$\text{UCB}_i(t) = \underbrace{\hat{\mu}_i(t)}_{\text{Exploitation Term}} + \underbrace{c \sqrt{\frac{\ln(t)}{N_i(t)}}}_{\text{Exploration Term}}$$

### Component Explanation:
1. **$\hat{\mu}_i(t)$ (Exploitation Term):** The empirical sample mean reward obtained by advertisement $i$. Higher observed CTR directly increases the score.
2. **$c$ (Exploration Coefficient):** Hyperparameter scaling the confidence interval width. In theoretical proofs, $c = \sqrt{2} \approx 1.414$ or $c = 2.0$. Higher values enforce broader exploration; lower values favor exploitation.
3. **$\ln(t)$ (Temporal Growth):** The natural logarithm of total elapsed simulation rounds across all advertisements. Because $\ln(t)$ increases monotonically, arms that are left unselected experience a steadily growing exploration bonus, preventing any arm from being permanently neglected.
4. **$N_i(t)$ (Sample Count):** The number of times advertisement $i$ has been displayed. Because $N_i(t)$ appears in the denominator, frequently selected arms experience a shrinking confidence interval, reflecting growing statistical certainty.

## 4.7 UCB Decision and Execution Process
The formal decision process of the UCB1 algorithm executes sequentially as follows:
1. **Initialization Phase ($t = 1, 2, \dots, K$):**
   - For the first $K$ rounds, select each advertisement $i \in \{1, \dots, K\}$ exactly once.
   - Observe initial reward $r_i \in \{0, 1\}$.
   - Set $N_i = 1$, $R_i = r_i$, and $\hat{\mu}_i = r_i$.
   - This cold-start initialization guarantees $N_i \ge 1$, avoiding mathematical division by zero in subsequent rounds.
2. **Optimization Loop ($t = K + 1, K + 2, \dots, T$):**
   - For each candidate advertisement $i \in \{1, \dots, K\}$, compute its upper confidence bound:
     $$\text{UCB}_i(t) = \hat{\mu}_i + c \sqrt{\frac{\ln(t)}{N_i}}$$
   - Select the advertisement with the maximal upper confidence bound:
     $$a_t = \arg\max_{i \in \{1, \dots, K\}} \text{UCB}_i(t)$$
     *(Ties are broken deterministically or uniformly at random).*
   - Display advertisement $a_t$ to the arriving user and observe binary response $r_t \sim \text{Bernoulli}(\mu_{a_t})$.
   - Update state parameters:
     $$N_{a_t} \leftarrow N_{a_t} + 1$$
     $$R_{a_t} \leftarrow R_{a_t} + r_t$$
     $$\hat{\mu}_{a_t} \leftarrow \frac{R_{a_t}}{N_{a_t}}$$
   - Increment elapsed rounds: $t \leftarrow t + 1$.
   - Repeat until horizon $T$ is reached.

## 4.8 Theoretical Regret and the Lai-Robbins Lower Bound
Regret quantifies the cumulative opportunity loss incurred by the agent due to lack of prior knowledge regarding which advertisement was truly optimal.

Let $\mu^* = \max_{i \in \{1, \dots, K\}} \mu_i$ denote the optimal expected CTR among all candidate advertisements. For each advertisement $i$, the sub-optimality gap is defined as:
$$\Delta_i = \mu^* - \mu_i$$
Note that for the optimal arm $i^*$, $\Delta_{i^*} = 0$, whereas for all suboptimal arms $i \neq i^*$, $\Delta_i > 0$.

### Instantaneous Expected Regret:
At round $t$, selecting action $a_t$ incurs instantaneous expected regret:
$$\delta_t = \mu^* - \mu_{a_t} = \Delta_{a_t}$$

### Cumulative Expected Regret:
Over a simulation horizon of $T$ rounds, the total cumulative expected regret $R(T)$ is the sum of instantaneous losses:
$$R(T) = \sum_{t=1}^T (\mu^* - \mu_{a_t}) = T \mu^* - \sum_{t=1}^T \mu_{a_t} = \sum_{i=1}^K \mathbb{E}[N_i(T)] \Delta_i$$

### Finite-Time Regret Bound of UCB1:
Auer et al. (2002) proved that for any $T > 1$ and Bernoulli reward distributions, the expected number of pulls for any suboptimal arm $i$ under UCB1 is bounded by [1]:
$$\mathbb{E}[N_i(T)] \le \frac{8 \ln(T)}{\Delta_i^2} + \left(1 + \frac{\pi^2}{3}\right)$$
Consequently, the cumulative expected regret of UCB1 satisfies:
$$R(T) \le \sum_{i: \mu_i < \mu^*} \left(\frac{8 \ln(T)}{\Delta_i} + \left(1 + \frac{\pi^2}{3}\right) \Delta_i\right) = O(\ln T)$$

This proof establishes that regret under UCB1 grows **logarithmically** with time horizon $T$. Because the derivative of $\ln(T)$ is $1/T \to 0$, the average regret per round vanishes asymptotically:
$$\lim_{T \to \infty} \frac{R(T)}{T} = 0$$
Algorithms exhibiting this property are termed **asymptotically zero-regret** policies, proving that UCB1 will converge on the winning ad and suffer negligible long-term loss.

<div style="page-break-after: always;"></div>

---

# CHAPTER 5 — SYSTEM DESIGN

## 5.1 System Architecture
The system architecture follows a clean, modular closed-loop design pattern separating environment simulation, algorithmic policy execution, metric logging, and user interface presentation. Figure 5.1 depicts the high-level architecture.

```text
+---------------------------------------------------------------------------------+
|                       USER INTERFACE (Streamlit Dashboard)                      |
|  +--------------------+  +----------------------+  +-------------------------+  |
|  | Single-Agent UCB1  |  | Multi-Agent Arena    |  | Contextual LinUCB       |  |
|  +--------------------+  +----------------------+  +-------------------------+  |
|  | Non-Stationary UCB |  | Offline Replay Eval  |  | Methodology & Viva Guide|  |
|  +--------------------+  +----------------------+  +-------------------------+  |
+----------------------------------------+----------------------------------------+
                                         | Explicit Button Triggers
                                         v
+---------------------------------------------------------------------------------+
|                      SIMULATION ENGINE (`src/simulation.py`)                    |
|  - Closed-Loop Interaction Harness                                              |
|  - High-Speed Pre-allocated Trajectory Buffers                                  |
|  - Reproducible Seeding via NumPy Generator                                     |
+-------------------+-----------------------------------------+-------------------+
                    |                                         |
         Pulls Arm  |                                  Action | Selected
                    v                                         v
+------------------------------------+   +----------------------------------------+
|       SIMULATED ENVIRONMENT        |   |           BANDIT AGENT CORE            |
|       (`src/environment.py`)       |   |          (`src/ucb_agent.py`)          |
|  - Hidden Ground Truth CTRs        |   |  - Counts N_i, Total Rewards R_i       |
|  - Bernoulli Sampling Generator    |   |  - Empirical Means \hat{\mu}_i         |
|  - Generates Binary Reward r_t     |   |  - Hoeffding Confidence Bound Bonus    |
+-------------------+----------------+   +--------------------+-------------------+
                    |                                         ^
                    | Observes Feedback (r_t in {0, 1})       |
                    +-----------------------------------------+
                                         |
                                         v
+---------------------------------------------------------------------------------+
|                       EVALUATION & VISUALIZATION ENGINE                         |
|                            (`src/evaluation.py`)                                |
|  - Metrics: Total Clicks, Average CTR, Cumulative Expected Regret               |
|  - Matplotlib Academic Publication Plots (Spine stripping, subtle grid)        |
|  - Tabular Summaries: Selections, Estimation Absolute Error, Allocation %       |
+---------------------------------------------------------------------------------+
```
**Figure 5.1: System Architecture of the Dynamic Advertisement Selection Framework**

## 5.2 System Workflow
The operational workflow proceeds through five sequential phases:
1. **Configuration Phase:** The user establishes simulation parameters (number of rounds $T$, exploration coefficient $c$, random seed, and hidden ground-truth CTR distributions).
2. **Cold-Start Arm Initialization:** The simulation engine triggers the agent to sample each candidate advertisement once to initialize empirical counters.
3. **Closed-Loop Optimization Loop:** For each subsequent round $t \in [K+1, T]$:
   - Agent computes UCB scores using current empirical means and uncertainty radiuses.
   - Agent selects arm $a_t = \arg\max_i \text{UCB}_i(t)$.
   - Environment evaluates $a_t$ and returns stochastic binary click $r_t \sim \text{Bernoulli}(\text{CTR}_{a_t})$.
   - Agent updates internal counters in $O(1)$ time.
   - Simulation engine logs instantaneous reward and regret to pre-allocated arrays.
4. **Trajectory Compilation:** Trajectories are packaged into a structured Pandas DataFrame.
5. **Analytics & Visualization:** Summary metrics are computed, tabular summaries generated, and publication-grade Matplotlib plots rendered to the dashboard.

## 5.3 Module Architecture
The codebase is structured into self-contained, decoupled Python modules within `src/`:
- **`src/base_agent.py`:** Defines the abstract base class `BaseBanditAgent`, establishing standard contracts (`select_ad`, `update`, `reset`, `get_estimated_ctrs`, `get_selection_counts`) to ensure seamless interchangeability across simulation harnesses.
- **`src/environment.py`:** Implements `AdvertisementEnvironment`, managing hidden ground-truth CTR arrays, input validation, and stochastic Bernoulli click generation.
- **`src/ucb_agent.py`:** Implements `UCB1Agent`, providing vector-accelerated UCB score computation, cold-start unvisited arm prioritization, and state update mechanisms.
- **`src/baseline_agents.py`:** Implements benchmark algorithms including `RandomAgent`, `EpsilonGreedyAgent` (with optional $O(K/t)$ decay schedules), and `ThompsonSamplingAgent` (Beta-Bernoulli conjugate distributions).
- **`src/contextual_environment.py` & `src/contextual_agent.py`:** Implements `ContextualAdvertisementEnvironment` and `LinUCBAgent`, modeling 5-dimensional user feature vectors and disjoint ridge regression updates.
- **`src/non_stationary.py`:** Implements `NonStationaryEnvironment`, `DiscountedUCBAgent`, and `SlidingWindowUCBAgent` for time-varying CTR shifts and ad fatigue modeling.
- **`src/offline_evaluator.py`:** Implements the Li et al. (WWW 2010) offline policy replay evaluator on historical impression logs.
- **`src/simulation.py`:** Coordinates vectorized closed-loop execution harnesses for single-agent, multi-agent, contextual, and non-stationary simulations.
- **`src/evaluation.py`:** Calculates summary KPI metrics and renders publication-quality Matplotlib figures.
- **`dashboard/app.py`:** Houses the multi-tab interactive Streamlit research environment.

## 5.4 Data Flow
Data flows through the system in an unbroken closed loop:
```text
(Input Parameters) ---> [Simulation Engine]
                              |
                     +--------+--------+
                     |                 |
                     v                 v
            [UCB1 Decision Rule]   [Bernoulli Environment]
                     |                 |
                     | Selected Arm    | Click Feedback
                     +-------->--------+
                               |
                        [State Updates]
                               |
                     [Trajectory Buffer]
                               |
                  [Pandas DataFrame Logs]
                               |
                     +---------+---------+
                     |                   |
                     v                   v
             [Summary Metrics]   [Matplotlib Visualizations]
                     |                   |
                     +---------+---------+
                               |
                               v
                     [Streamlit Dashboard]
```

## 5.5 User Interface Architecture
The front-end is engineered in Streamlit with an academic research philosophy:
- **Max-Width Workspace Layout:** The interface is constrained to an 1120px centered container, preventing horizontal dead space on high-resolution displays.
- **Strict State Separation:** Interactive inputs alter only a working Configuration State. Executed results and visualizations update *only* when the user explicitly clicks an action button (e.g., **Run Simulation**, **Run Benchmark Arena**, **Run LinUCB Personalization**).
- **Configuration Mutation Banner:** If an input parameter is modified while viewing existing results, the previous results remain visible and a contextual badge alerts the user: *"Configuration changed. Click Run Simulation to update results."*
- **Modular Tab Navigation:** Six independent research workspaces: (1) Single-Agent UCB1, (2) Algorithm Arena, (3) Contextual Personalization, (4) Non-Stationary Dynamics, (5) Offline Evaluation, and (6) Methodology & Viva Guide.

<div style="page-break-after: always;"></div>

---

# CHAPTER 6 — IMPLEMENTATION

## 6.1 Development Environment
The software was authored and verified using:
- **Operating System:** Windows 11 (64-bit), x86_64 architecture.
- **Interpreter:** Python 3.11.9.
- **Code Editor / IDE:** Visual Studio Code with Python Language Server and Black formatter.
- **Virtual Environment:** Python standard `venv` managing library dependencies.
- **Version Control:** Git version 2.45+ tracking history on GitHub.

## 6.2 Advertisement Environment Module
The `AdvertisementEnvironment` class in `src/environment.py` simulates user exposure to candidate advertisements.

```python
class AdvertisementEnvironment:
    def __init__(self, true_ctrs=None, ad_names=None, seed=None):
        if true_ctrs is None:
            true_ctrs = [0.05, 0.10, 0.03, 0.15, 0.08]
        self.true_ctrs = np.array(true_ctrs, dtype=np.float64)
        self.num_ads = len(self.true_ctrs)
        self.ad_names = ad_names or [f"Ad {i+1}" for i in range(self.num_ads)]
        self._rng = np.random.default_rng(seed)

    def pull_ad(self, ad_index: int) -> int:
        if not (0 <= ad_index < self.num_ads):
            raise IndexError("ad_index out of bounds.")
        click_prob = self.true_ctrs[ad_index]
        return int(self._rng.random() < click_prob)
```
Input validation ensures all true CTR values reside within $[0.0, 1.0]$. The `_rng` object encapsulates a modern PCG64 random number generator, guaranteeing statistical independence and repeatability.

## 6.3 UCB1 Agent Implementation
The `UCB1Agent` class in `src/ucb_agent.py` implements the core bandit logic inheriting from `BaseBanditAgent`.

### Algorithmic Pseudocode:
```text
Algorithm: UCB1 Agent Decision and Update
Input: Number of arms K, Exploration parameter c > 0

Initialization:
    counts[1 ... K] <- 0
    total_rewards[1 ... K] <- 0.0
    estimated_ctrs[1 ... K] <- 0.0
    current_round <- 0

Function select_ad():
    // Step 1: Cold-start check
    For each arm i in {1 ... K}:
        If counts[i] == 0:
            Return i
    
    // Step 2: Compute UCB scores
    t_next <- current_round + 1
    For each arm i in {1 ... K}:
        exploration_bonus <- c * sqrt(ln(t_next) / counts[i])
        ucb[i] <- estimated_ctrs[i] + exploration_bonus
        
    Return argmax_i (ucb[i])

Function update(ad_index, reward):
    current_round <- current_round + 1
    counts[ad_index] <- counts[ad_index] + 1
    total_rewards[ad_index] <- total_rewards[ad_index] + reward
    estimated_ctrs[ad_index] <- total_rewards[ad_index] / counts[ad_index]
```

## 6.4 Closed-Loop Simulation Engine
The simulation execution function `run_simulation()` in `src/simulation.py` handles high-throughput closed-loop execution. To optimize runtime performance, all trajectory arrays (`selected_ads`, `rewards`, `cumulative_rewards`, `inst_expected_regrets`, `cum_expected_regrets`) are pre-allocated with fixed NumPy buffer sizes rather than dynamically appending to lists. A 10,000-round simulation completes in under 120 milliseconds.

## 6.5 Reward and Metric Calculations
At each round $t$, the simulation harness logs:
1. Observed Reward $r_t \in \{0, 1\}$.
2. Cumulative Reward: $\sum_{s=1}^t r_s$.
3. System Click-Through Rate: $\text{CTR}(t) = \frac{\sum_{s=1}^t r_s}{t}$.

## 6.6 Empirical CTR Estimation
The empirical CTR $\hat{\mu}_i$ is updated recursively in $O(1)$ time whenever arm $i$ is pulled:
$$\hat{\mu}_i(t) = \frac{R_i(t)}{N_i(t)}$$
The estimation error relative to ground truth is monitored as:
$$\text{Error}_i(t) = |\hat{\mu}_i(t) - \mu_i|$$

## 6.7 Regret Tracking
To evaluate algorithm efficiency against an omniscient oracle, the engine logs theoretical expected regret:
$$\delta_t = \mu^* - \mu_{a_t}$$
$$\text{Cumulative Regret}(t) = \sum_{s=1}^t (\mu^* - \mu_{a_s})$$
where $\mu^* = \max_i \mu_i = 0.15$ for the baseline configuration.

## 6.8 Multi-Agent Benchmark Arena
The benchmark harness `run_multi_agent_simulation()` instantiates five distinct agent architectures:
1. `UCB1Agent`: Classical Upper Confidence Bound with $c = 2.0$.
2. `ThompsonSamplingAgent`: Posterior sampling using independent $\text{Beta}(\alpha_i, \beta_i)$ distributions with uninformative uniform priors $\alpha_0 = 1, \beta_0 = 1$.
3. `EpsilonGreedyAgent (decay)`: Decaying exploration schedule $\varepsilon_t = \min(1.0, \frac{\varepsilon_0 K}{t})$ with $\varepsilon_0 = 0.5$.
4. `EpsilonGreedyAgent (eps=0.1)`: Constant exploration rate $\varepsilon = 0.1$.
5. `RandomAgent`: Uniform random selection $a_t \sim \mathcal{U}\{0, K-1\}$, modeling non-adaptive A/B testing.

Crucially, each agent is evaluated against an identical stochastic sequence generated by resetting the environment with the same seed ($\text{seed} = 42$), guaranteeing fair comparison.

## 6.9 Contextual Personalization (LinUCB)
In `src/contextual_agent.py`, the `LinUCBAgent` implements Disjoint Linear UCB [6]. For each arm $a \in \{1, \dots, K\}$:
- The design covariance matrix $A_a \in \mathbb{R}^{5 \times 5}$ is initialized to the identity matrix $I_5$.
- The response bias vector $b_a \in \mathbb{R}^5$ is initialized to $\mathbf{0}$.
- For an incoming normalized user feature vector $x_t \in \mathbb{R}^5$:
  $$\hat{\theta}_a = A_a^{-1} b_a$$
  $$\text{UCB}_a(x_t) = x_t^T \hat{\theta}_a + \alpha \sqrt{x_t^T A_a^{-1} x_t}$$
- Upon observing click $r_t$, online ridge regression parameters update in $O(d^2)$ time:
  $$A_{a_t} \leftarrow A_{a_t} + x_t x_t^T, \quad b_{a_t} \leftarrow b_{a_t} + r_t x_t$$

The user context vector comprises 5 binary and continuous dimensions:
$$x = [\text{Bias}, \text{Mobile Device}, \text{Age Under 30}, \text{Evening Browsing}, \text{High Tech Affinity}]^T$$

## 6.10 Non-Stationary Dynamics Engine
In `src/non_stationary.py`, the `NonStationaryEnvironment` simulates sudden ad fatigue and market shifts. At a configurable changepoint $t_{\text{change}} = 5,000$, the true CTR of the previously optimal Ad 4 collapses from 15% to 2%, while Ad 2 surges from 10% to 18%.
- `SlidingWindowUCBAgent` maintains a FIFO deque of historical observations of window size $W = 1,500$.
- `DiscountedUCBAgent` discounts historical observations by $\gamma = 0.998$ per round:
  $$N_i(\gamma, t) = \sum_{s=1}^t \gamma^{t-s} \mathbb{I}(a_s = i), \quad R_i(\gamma, t) = \sum_{s=1}^t \gamma^{t-s} r_s \mathbb{I}(a_s = i)$$

## 6.11 Unbiased Offline Policy Evaluator
In `src/offline_evaluator.py`, the replay method proposed by Li et al. (WWW 2010) is implemented to evaluate policies on 10,000 historical logged impressions [6]:
1. A stream of historical impressions $\{(x_t, a_t, r_t)\}_{t=1}^S$ generated by a uniform random logging policy is ingested.
2. For each historical event, the candidate policy proposes action $\hat{a}_t$.
3. If $\hat{a}_t = a_t$ (match), the event is **accepted**: the agent updates its policy using historical reward $r_t$, and the accepted impression count increments.
4. If $\hat{a}_t \neq a_t$, the event is **rejected** and discarded without state modification.
5. Li et al. proved that the effective CTR on accepted events:
   $$\text{CTR}_{\text{offline}} = \frac{\sum_{t \in \text{Accepted}} r_t}{|\text{Accepted}|}$$
   is an **unbiased estimator** of the policy's true expected online payoff.

## 6.12 Interactive Research Dashboard
The Streamlit application (`dashboard/app.py`) provides:
- Metric KPI cards showing harvested clicks, system CTR, theoretical regret, and winning ad identification.
- Ground Truth CTR parameter grid allowing users to configure custom environments.
- Responsive tabular reports displaying per-ad selection counts, allocation percentages, empirical CTR estimates, and absolute errors.
- Publication-quality Matplotlib figures rendered directly via `st.pyplot()`.

<div style="page-break-after: always;"></div>

---

# CHAPTER 7 — EXPERIMENTAL METHODOLOGY

## 7.1 Experimental Setup
To validate the system, experiments were conducted using the standardized parameter configurations specified in Table 7.1.

**Table 7.1: Experimental Simulation Configuration Parameters**

| Parameter | Configuration Setting | Justification |
|:---|:---|:---|
| **Candidate Advertisements ($K$)** | 5 Arms | Represents a standard ad inventory slot. |
| **Advertisement Identifiers** | Ad 1, Ad 2, Ad 3, Ad 4, Ad 5 | Clear categorical labeling. |
| **Ground Truth CTRs ($\mu$)** | $[0.05, 0.10, 0.03, 0.15, 0.08]$ | Reflects typical industry conversion spread; Ad 4 is optimal ($\mu^* = 0.15$). |
| **Simulation Horizon ($T$)** | 10,000 Rounds (Impressions) | Sufficient sample volume to demonstrate asymptotic convergence. |
| **Exploration Coefficient ($c$)** | 2.0 | Standard theoretical exploration parameter for UCB1. |
| **Random Number Generator** | NumPy PCG64 (`seed = 42`) | Ensures complete reproducibility across benchmark runs. |
| **LinUCB Alpha ($\alpha$)** | 1.0 | Standard confidence ellipsoid radius for ridge regression. |
| **Non-Stationary Changepoint** | Round $t = 5,000$ | Divides the horizon into equal pre-shift and post-shift phases. |
| **Post-Shift CTRs** | $[0.05, 0.18, 0.03, 0.02, 0.08]$ | Ad 4 collapses (15% $\to$ 2%); Ad 2 surges (10% $\to$ 18%). |

## 7.2 Experimental Procedure
1. Initialize the `AdvertisementEnvironment` with true CTR vector $[0.05, 0.10, 0.03, 0.15, 0.08]$ and seed 42.
2. Instantiate `UCB1Agent` with $K = 5$ and $c = 2.0$.
3. Execute `run_simulation()` for $T = 10,000$ rounds.
4. Record trajectory data, compute summary statistics, and generate single-agent analytical figures.
5. In the Multi-Agent Arena, instantiate the five candidate algorithms and execute `run_multi_agent_simulation()` over identical 10,000-round sequences.
6. In the Contextual module, simulate 10,000 user arrivals with 5-dimensional context vectors and run `run_contextual_simulation()`.
7. In the Non-Stationary module, execute `run_non_stationary_simulation()` introducing the abrupt shift at round 5,000.
8. In the Offline module, generate 10,000 logged impression events and execute `evaluate_policy_offline_replay()`.

## 7.3 Evaluation Metrics
- **Total Reward (Clicks):** Total clicks accumulated over horizon $T$: $\sum_{t=1}^T r_t$.
- **System Click-Through Rate:** Empirical conversion efficiency: $\text{CTR} = \frac{\text{Total Reward}}{T}$.
- **Cumulative Expected Regret:** Theoretical opportunity loss: $\sum_{t=1}^T (\mu^* - \mu_{a_t})$.
- **Arm Allocation Percentage:** Percentage of total impressions routed to arm $i$: $\frac{N_i(T)}{T} \times 100\%$.
- **Estimation Absolute Error:** Absolute error between estimated CTR and ground truth: $|\hat{\mu}_i - \mu_i|$.
- **Relative Click Lift:** Percentage increase in harvested clicks relative to the random baseline:
  $$\text{Lift} = \frac{\text{Clicks}_{\text{agent}} - \text{Clicks}_{\text{random}}}{\text{Clicks}_{\text{random}}} \times 100\%$$

## 7.4 Baseline Algorithms
1. **Random / Static A/B Testing:** Uniform random sampling across arms at every step ($p_i = 1/K$). Models equal-traffic A/B allocation.
2. **Fixed $\varepsilon$-Greedy ($\varepsilon = 0.1$):** Exploits current best empirical arm 90% of the time, explores randomly 10% of the time.
3. **Decaying $\varepsilon$-Greedy:** Modulates exploration as $\varepsilon_t = \min(1.0, 2.5 / t)$, reducing exploration as rounds progress.
4. **Thompson Sampling:** Samples candidate CTRs from posterior Beta distributions $\text{Beta}(\alpha_i, \beta_i)$ at every step.

## 7.5 Reproducibility and Determinism
By explicitly seeding the PCG64 generator (`seed = 42`), every Bernoulli trial and random arm selection is deterministic. Re-running the simulation on any POSIX or Windows machine yields identical numbers.

<div style="page-break-after: always;"></div>

---

# CHAPTER 8 — RESULTS AND ANALYSIS

## 8.1 Single-Agent UCB1 Performance
The single-agent UCB1 simulation was executed across $T = 10,000$ rounds with exploration parameter $c = 2.0$ and random seed 42. Table 8.1 reports the complete verified empirical summary.

**Table 8.1: Single-Agent UCB1 Simulation Summary ($T = 10,000$ Rounds, Seed = 42)**

| Advertisement | Ground Truth CTR ($\mu_i$) | Empirical Estimated CTR ($\hat{\mu}_i$) | Absolute Error ($|\hat{\mu}_i - \mu_i|$) | Total Selections ($N_i$) | Allocation % | Total Clicks Harvested ($R_i$) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Ad 1** | 0.0500 | 0.052980 | 0.002980 | 1,057 | 10.57% | 56 |
| **Ad 2** | 0.1000 | 0.095480 | 0.004520 | 1,770 | 17.70% | 169 |
| **Ad 3** | 0.0300 | 0.032596 | 0.002596 | 859 | 8.59% | 28 |
| **Ad 4 (Optimal)** | **0.1500** | **0.155547** | **0.005547** | **5,156** | **51.56%** | **802** |
| **Ad 5** | 0.0800 | 0.061313 | 0.018687 | 1,158 | 11.58% | 71 |
| **System Total** | — | — | — | **10,000** | **100.00%** | **1,126** |

### Key Overall Metrics:
- **Total Clicks Harvested:** 1,126 clicks.
- **Overall System CTR:** 0.1126 (11.26%).
- **Cumulative Theoretical Regret:** 378.34.
- **Optimal Ad Identified by Agent:** Ad 4 (Selected 5,156 times, accounting for 71.2% of all harvested clicks).

## 8.2 Advertisement Selection Distribution
As documented in Table 8.1, UCB1 routed **5,156 out of 10,000 impressions (51.56%)** to the truly optimal creative, Ad 4. The second most frequently selected creative was Ad 2 (1,770 impressions, 17.70%), which possessed the second-highest ground truth CTR (10.0%). The lowest-performing creative, Ad 3 (CTR = 3.0%), received the fewest impressions (859 impressions, 8.59%). 

This distribution precisely mirrors the mathematical requirements of the Lai-Robbins theorem: suboptimal arms with smaller gaps $\Delta_i$ (such as Ad 2, $\Delta_2 = 0.05$) require more samples to distinguish from the optimal arm than arms with large gaps (such as Ad 3, $\Delta_3 = 0.12$).

## 8.3 CTR Convergence and Estimation Accuracy
The absolute estimation error across all arms remained exceptionally low, ranging from 0.002596 (Ad 3) to 0.018687 (Ad 5). Ad 4 converged to an empirical estimate of 0.155547 against its ground truth of 0.150000 (an error of just 0.0055). This validates that the UCB1 confidence bound provides sufficient exploration to estimate arm reward probabilities accurately without over-allocating traffic.

## 8.4 Cumulative Reward Trajectory
During the initial 1,000 rounds, the cumulative reward trajectory displays an exploratory curvature as the agent tests all five arms to distinguish their means. Once Ad 4 establishes statistical superiority (around round 2,000), the curve transitions into a steep, constant linear trajectory with a slope approaching $\mu^* = 0.15$, confirming that the agent exploits the optimal arm for the vast majority of subsequent traffic.

## 8.5 Cumulative Regret Analysis
The cumulative regret curve demonstrates the classic sublinear logarithmic trajectory $O(\ln T)$ predicted by Auer et al. (2002) [1]. Regret accumulates rapidly during the initial 1,500 exploration rounds, but rapidly decelerates as $t$ increases. Between rounds 5,000 and 10,000, cumulative regret increases by only 72 units, demonstrating an asymptotically flattening slope that contrasts sharply with the steep linear growth of static testing.

## 8.6 Multi-Algorithm Benchmark Comparison
Table 8.2 summarizes the comparative performance of the five benchmarked algorithms over $T = 10,000$ rounds under identical random seeding.

**Table 8.2: Multi-Algorithm Benchmark Arena Leaderboard ($T = 10,000$ Rounds, Seed = 42)**

| Rank | Algorithm Policy | Total Clicks Harvested | System CTR | Cumulative Regret | Regret Complexity | Relative Click Lift vs. Static Random |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| 🥇 | **$\varepsilon$-Greedy (decay schedule)** | **1,491** | **14.91%** | **10.22** | Sublinear $O(\ln T)$ | **+85.0%** |
| 🥈 | **Thompson Sampling (Beta Prior)** | **1,464** | **14.64%** | **31.77** | Sublinear $O(\ln T)$ | **+81.6%** |
| 🥉 | **$\varepsilon$-Greedy (fixed $\varepsilon = 0.1$)** | **1,429** | **14.29%** | **71.53** | Linear $O(\varepsilon T)$ | **+77.3%** |
| 4 | **UCB1 ($c = 2.0$)** | **1,126** | **11.26%** | **378.34** | Sublinear $O(\ln T)$ | **+39.7%** |
| 5 | **Random / Static A/B Testing** | **806** | **8.06%** | **686.96** | Linear $O(T)$ | **Baseline (0.0%)** |

### Benchmark Insights:
1. **Dramatic Click Lift Over Static A/B Testing:** All four reinforcement learning policies significantly outperformed static random allocation. Decaying $\varepsilon$-greedy and Thompson Sampling delivered **1,491 clicks (+85.0% lift)** and **1,464 clicks (+81.6% lift)** respectively, compared to just 806 clicks for static testing.
2. **Thompson Sampling vs. UCB1:** Thompson Sampling outperformed UCB1 (1,464 vs. 1,126 clicks; regret 31.77 vs. 378.34). This empirical phenomenon is well-documented in literature (Chapelle & Li, 2011): UCB1's Hoeffding bound represents a distribution-free worst-case guarantee, making its exploration bonus relatively conservative ($c = 2.0$), whereas Thompson Sampling's posterior draws explore more efficiently in Bernoulli settings [8].
3. **Decaying vs. Fixed $\varepsilon$-Greedy:** Fixed $\varepsilon$-greedy ($\varepsilon = 0.1$) accumulated 71.53 regret due to persistent 10% random exploration, whereas decaying $\varepsilon$-greedy successfully reduced exploration to achieve a leaderboard-topping regret of 10.22.

## 8.7 Contextual LinUCB Results
Table 8.3 reports the results of the Disjoint LinUCB algorithm evaluated across 10,000 user visits conditioned on 5-dimensional user feature vectors.

**Table 8.3: Contextual LinUCB Performance Metrics ($T = 10,000$ Rounds, Seed = 42)**

| Metric Name | Value |
|:---|:---:|
| **Total Contextual Clicks Harvested** | **6,563** |
| **Personalized System CTR** | **65.63%** |
| **Cumulative Contextual Regret** | **248.19** |
| **Context Feature Dimensions ($d$)** | 5 |
| **Candidate Personalized Creatives** | 4 |

By conditioning ad selection on user demographics, LinUCB achieved an extraordinary **65.63% CTR**, far exceeding the maximum possible CTR of any single static ad arm (15%). The agent learned to display Gaming ads to young mobile visitors, Cloud SaaS ads to technical desktop users, and Fashion ads during evening hours, demonstrating the immense economic value of contextual personalization.

## 8.8 Non-Stationary Results
Table 8.4 documents algorithm adaptation when an abrupt regime shift occurred at round $t = 5,000$ (Ad 4 collapsed from 15% $\to$ 2%, while Ad 2 surged from 10% $\to$ 18%).

**Table 8.4: Non-Stationary Regime Shift Performance Metrics ($T = 10,000$ Rounds, Changepoint = 5,000)**

| Algorithm Policy | Final Cumulative Regret | Regret Reduction vs. Stationary UCB1 | Adaptation Characteristics |
|:---|:---:|:---:|:---|
| **Stationary UCB1** | 149.10 | Baseline (0.0%) | Sluggish adaptation; high historical pull count ($N_4$) delays exploration of new winner. |
| **Discounted-UCB (D-UCB, $\gamma = 0.998$)** | 280.96 | -88.4% | Higher total regret due to continuous variance amplification across the long horizon. |
| **Sliding-Window UCB (SW-UCB, $W = 1,500$)** | **128.54** | **+13.8%** | **Fastest adaptation; completely purges obsolete pre-shift observations.** |

Sliding-Window UCB achieved a **13.8% reduction in cumulative regret** relative to stationary UCB1. By discarding observations older than 1,500 rounds, SW-UCB rapidly reduced the perceived value of the decayed Ad 4 and transitioned traffic to the newly surging Ad 2.

## 8.9 Offline Evaluation Results
Table 8.5 presents the results of the Li et al. (WWW 2010) offline policy replay evaluation executed across 10,000 logged historical impression events.

**Table 8.5: Unbiased Offline Policy Replay Evaluation on Historical Logs (10,000 Samples)**

| Evaluated Policy | Accepted Impressions ($T_{\text{acc}}$) | Total Clicks | Effective Replay CTR ($\hat{\mu}_{\text{offline}}$) |
|:---|:---:|:---:|:---:|
| **UCB1 ($c = 1.5$)** | 2,506 | 987 | 39.39% |
| **Thompson Sampling** | 2,468 | 1,051 | 42.59% |
| **Contextual LinUCB** | 2,502 | 1,199 | **47.92%** |

Because the historical logging policy was uniform random across 4 arms, each target policy accepted approximately 25% of the logged stream ($\approx 2,500$ events). LinUCB achieved the highest effective offline CTR at **47.92%**, confirming its superiority under counterfactual evaluation on logged historical data.

## 8.10 Overall Discussion
The experimental findings provide rigorous empirical validation of multi-armed bandit theory:
1. **Logarithmic Regret Verification:** Single-agent UCB1 demonstrated clear sublinear regret growth, validating Auer et al.'s theoretical upper bound.
2. **Superiority Over Static Allocation:** Adaptive allocation harvested 39.7% to 85.0% more clicks than static A/B testing over 10,000 rounds, illustrating the enormous economic opportunity cost eliminated by Reinforcement Learning.
3. **Bayesian vs. Frequentist Trade-Offs:** While UCB1 provides deterministic worst-case guarantees, Thompson Sampling delivers superior empirical performance in Bernoulli settings due to smoother posterior sampling.

<div style="page-break-after: always;"></div>

---

# CHAPTER 9 — SYSTEM VERIFICATION AND TESTING

## 9.1 Testing Strategy
A multi-tier automated testing strategy was implemented across unit tests, mathematical convergence tests, simulation boundary checks, and headless UI interaction tests:
- **Unit Testing:** Validates individual class contracts, argument bounds, and exception handling.
- **Mathematical Testing:** Asserts that confidence bounds, regret calculations, and empirical mean estimators conform to formal equations.
- **Integration Testing:** Executes full multi-round simulation pipelines to verify data frame integrity and absence of memory leaks.
- **UI State Management Testing:** Uses Streamlit's native `AppTest` framework to verify that input modifications do NOT trigger automatic simulations and that buttons properly control execution.

## 9.2 Functional and Algorithmic Test Cases
Table 9.1 details the verified automated test suite consisting of 42 test cases across 8 test suites, all passing with 100% success.

**Table 9.1: Automated System Verification and Test Suite Results**

| Test Suite Module | Test Scope / Focus | Test Count | Pass Rate | Execution Time |
|:---|:---|:---:|:---:|:---:|
| `test_ucb_agent.py` | UCB1 calculation, cold-start initialization, arm updates, reset | 6 | 100% (6/6) | 0.12s |
| `test_environment.py` | Bernoulli sampling, CTR bounds validation, best ad lookup | 5 | 100% (5/5) | 0.08s |
| `test_simulation.py` | Single-agent trajectory logging, dataframe column validation | 3 | 100% (3/3) | 0.35s |
| `test_baseline_agents.py` | Random, EpsilonGreedy (fixed/decay), Thompson Sampling posteriors | 8 | 100% (8/8) | 0.22s |
| `test_contextual.py` | LinUCB matrix dimensions, ridge regression updates, context bounds | 6 | 100% (6/6) | 0.18s |
| `test_non_stationary.py` | Changepoint triggering, D-UCB discounting, SW-UCB windowing | 5 | 100% (5/5) | 0.15s |
| `test_multi_agent.py` | Multi-agent arena harness, leaderboard sorting, seed determinism | 3 | 100% (3/3) | 0.28s |
| `test_ui_interaction.py` | Headless Streamlit AppTest: initial load, explicit run, reset, change notice | 6 | 100% (6/6) | 7.88s |
| **Total Test Suite** | **Comprehensive Full System Verification** | **42** | **100% (42/42)** | **9.26s** |

## 9.3 UCB Algorithm Unit Verification
Unit tests in `tests/test_ucb_agent.py` formally verified:
- Initial arm selection returns unvisited arms ($N_i = 0$) with UCB score $= \infty$.
- Score computation matches $\hat{\mu}_i + c \sqrt{\ln(t)/N_i}$ with precision $\epsilon < 10^{-7}$.
- Updating arm $i$ increments $N_i$, updates total reward $R_i$, and recalculates $\hat{\mu}_i$.
- Negative exploration parameters ($c \le 0$) raise `ValueError`.

## 9.4 Simulation Boundary and Parameter Stress Testing
Boundary tests verified:
- Horizon $T = 1$ correctly executes cold-start arm selection without crashing.
- Extreme CTR distributions ($[0.0, 0.0, 0.0, 1.0]$) converge within 20 rounds.
- Non-stationary changepoints beyond simulation horizon $T$ do not cause execution errors.

## 9.5 Dashboard and UI State Management Testing
Headless automated interaction tests in `tests/test_ui_interaction.py` verified:
- **Test 1 (Initial Load):** On startup, no simulation runs automatically; empty placeholder cards are shown.
- **Test 2 (No Auto-Run):** Modifying the rounds input from 10,000 to 20,000 does NOT execute a simulation; empty state persists.
- **Test 3 (Explicit Run):** Clicking **Run Simulation** executes UCB1 and renders metric cards, charts, and tables.
- **Test 4 (Change Notice):** Modifying parameter $c$ after a simulation retains existing results and displays the notice: *"Configuration changed. Click Run Simulation to update results."*
- **Test 5 (Reset):** Clicking **Reset** clears state and restores clean empty placeholders.
- **Test 6 (Multiple Runs):** Successive simulation executions update trajectory data without state corruption.

## 9.6 Defensive Error Handling
All modules incorporate defensive programming:
- `AdvertisementEnvironment`: Rejects empty CTR lists, non-numeric values, or CTRs outside $[0.0, 1.0]$.
- `UCB1Agent`: Validates $c > 0$ and checks array index bounds.
- `LinUCBAgent`: Validates context vector length against `context_dim` and uses `np.linalg.inv` with numerical clipping.

<div style="page-break-after: always;"></div>

---

# CHAPTER 10 — SYSTEM LIMITATIONS

While the system provides a comprehensive, rigorous implementation of multi-armed bandit algorithms, the following academic limitations should be noted:
1. **Synthetic User Feedback Distributions:** User click feedback is generated from stationary or piecewise-stationary Bernoulli trials. Real-world consumer behavior incorporates complex latent dynamics such as position bias, banner blindness, temporal seasonality, and network effects.
2. **Absence of Real-Time Ad-Exchange Auction Mechanics:** Production advertising systems operate through real-time bidding (RTB) auctions (e.g., First-Price and Generalized Second-Price auctions) where ad placement depends not only on CTR, but also on bid prices, publisher reserve floors, and competitor bidding dynamics.
3. **Discrete Impression Round Representation:** Simulations execute in discrete sequential steps ($t = 1, \dots, T$). In production ad networks, impressions occur concurrently across distributed server clusters with asynchronous feedback delays.
4. **Linear Contextual Feature Mapping:** The LinUCB implementation assumes expected rewards are linear in the context features ($x^T \theta$). In complex applications, user-creative interactions exhibit non-linear feature interactions requiring deep neural network representations.

<div style="page-break-after: always;"></div>

---

# CHAPTER 11 — FUTURE SCOPE AND ENHANCEMENTS

To extend this work into production-scale advertising infrastructure, future enhancements include:
1. **Evaluation on Industrial Ad Click Datasets:** Benchmark the algorithms against large-scale public advertising datasets, such as the Criteo 1TB Click Logs or Avazu Mobile Click datasets.
2. **Deep Neural Contextual Bandits:** Replace linear ridge regression with deep neural networks (e.g., Neural Linear Bandits or Deep Bayesian Bandits) to learn non-linear representations of rich text, image, and user features.
3. **Continuous Streaming and Low-Latency Distributed Serving:** Package the decision engine as a microservice using FastAPI or gRPC integrated with Redis for sub-millisecond feature lookup and Apache Kafka for asynchronous feedback ingestion.
4. **Budget-Constrained Bandits (Banded Knapsacks):** Extend the formulation to incorporate advertiser campaign budgets and cost-per-click (CPC) constraints, optimizing total revenue under knapsack budget constraints.
5. **Thompson Sampling for Contextual Bandits:** Implement Linear Thompson Sampling (LinTS) to provide Bayesian posterior sampling over high-dimensional contextual feature spaces.

<div style="page-break-after: always;"></div>

---

# CHAPTER 12 — CONCLUSION

In this project, the dynamic online advertisement selection challenge was formulated and resolved as a Multi-Armed Bandit Reinforcement Learning problem using the Upper Confidence Bound (UCB1) algorithm. Constructed entirely from foundational mathematical principles without black-box reinforcement learning libraries, the resulting system delivers an end-to-end research and experimentation platform.

Through closed-loop stochastic simulations across $T = 10,000$ rounds, the UCB1 algorithm was demonstrated to balance exploration and exploitation effectively, achieving an empirical CTR of 11.26% and sublinear logarithmic regret ($O(\ln T)$) while directing over 51% of impressions to the optimal creative. In multi-agent benchmark comparisons, adaptive reinforcement learning policies outperformed traditional static A/B testing by up to +85.0% in total harvested clicks. Extensions to contextual bandits via Disjoint LinUCB achieved an effective personalized CTR of 65.63%, while Sliding-Window UCB demonstrated a 13.8% regret reduction over stationary algorithms under abrupt market regime shifts. Unbiased offline policy replay verified these findings on historical logs.

Finally, an interactive research dashboard adhering to rigorous UI state separation provides intuitive controls, real-time persona simulation, and publication-quality visual analytics. Backed by an automated test suite with 100% pass verification across 42 test cases, this project provides a solid, verifiable, and academically rigorous foundation for reinforcement learning in online decision systems.

<div style="page-break-after: always;"></div>

---

# REFERENCES

[1] P. Auer, N. Cesa-Bianchi, and P. Fischer, "Finite-time analysis of the multiarmed bandit problem," *Machine Learning*, vol. 47, no. 2-3, pp. 235–256, 2002.

[2] T. L. Lai and H. Robbins, "Asymptotically efficient adaptive allocation rules," *Advances in Applied Mathematics*, vol. 6, no. 1, pp. 4–22, 1985.

[3] R. S. Sutton and A. G. Barto, *Reinforcement Learning: An Introduction*, 2nd ed. Cambridge, MA: The MIT Press, 2018.

[4] W. R. Thompson, "On the likelihood that one unknown probability exceeds another in view of the evidence of two samples," *Biometrika*, vol. 25, no. 3/4, pp. 285–294, 1933.

[5] S. Agrawal and N. Goyal, "Analysis of Thompson sampling for the multi-armed bandit problem," in *Proceedings of the 25th Annual Conference on Learning Theory (COLT)*, vol. 23, pp. 39.1–39.26, 2012.

[6] L. Li, W. Chu, J. Langford, and R. E. Schapire, "A contextual-bandit approach to personalized news headline recommendation," in *Proceedings of the 19th International Conference on World Wide Web (WWW)*, pp. 661–670, 2010.

[7] A. Garivier and E. Moulines, "On upper-confidence bound policies for switching bandit problems," in *Proceedings of the 22nd International Conference on Algorithmic Learning Theory (ALT)*, LNCS 6925, pp. 174–188, 2011.

[8] O. Chapelle and L. Li, "An empirical evaluation of Thompson sampling," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 24, pp. 2249–2257, 2011.

[9] T. Lattimore and C. Szepesvári, *Bandit Algorithms*. Cambridge: Cambridge University Press, 2020.

[10] A. Slivkins, "Introduction to multi-armed bandits," *Foundations and Trends in Machine Learning*, vol. 12, no. 1-2, pp. 1–286, 2019.

[11] L. Kocsis and C. Szepesvári, "Bandit based Monte-Carlo planning," in *Proceedings of the 17th European Conference on Machine Learning (ECML)*, pp. 282–293, 2006.

[12] W. Hoeffding, "Probability inequalities for sums of bounded random variables," *Journal of the American Statistical Association*, vol. 58, no. 301, pp. 13–30, 1963.

<div style="page-break-after: always;"></div>

---

# APPENDIX A — CORE IMPLEMENTATION SOURCE CODE

### A.1 Base Bandit Agent Interface (`src/base_agent.py`)
```python
from abc import ABC, abstractmethod
import numpy as np

class BaseBanditAgent(ABC):
    def __init__(self, num_ads: int, name: str = "BaseAgent") -> None:
        if num_ads < 1:
            raise ValueError(f"num_ads must be at least 1, got {num_ads}.")
        self.num_ads = num_ads
        self.name = name
        self.counts = np.zeros(num_ads, dtype=np.int64)
        self.total_rewards = np.zeros(num_ads, dtype=np.float64)
        self.estimated_ctrs = np.zeros(num_ads, dtype=np.float64)
        self.current_round = 0

    @abstractmethod
    def select_ad(self) -> int:
        pass

    def update(self, ad_index: int, reward: int) -> None:
        self.current_round += 1
        self.counts[ad_index] += 1
        self.total_rewards[ad_index] += reward
        self.estimated_ctrs[ad_index] = self.total_rewards[ad_index] / self.counts[ad_index]

    def reset(self) -> None:
        self.counts.fill(0)
        self.total_rewards.fill(0.0)
        self.estimated_ctrs.fill(0.0)
        self.current_round = 0
```

### A.2 UCB1 Agent Implementation (`src/ucb_agent.py`)
```python
import numpy as np
from src.base_agent import BaseBanditAgent

class UCB1Agent(BaseBanditAgent):
    def __init__(self, num_ads: int, c: float = 2.0, name: str = "UCB1") -> None:
        if c <= 0.0:
            raise ValueError(f"Exploration coefficient c must be positive, got {c}.")
        super().__init__(num_ads=num_ads, name=name)
        self.c = float(c)

    def calculate_ucb(self, ad_index: int, total_rounds: int = None) -> float:
        if self.counts[ad_index] == 0:
            return float(np.inf)
        t = total_rounds if total_rounds is not None else max(1, self.current_round)
        empirical_mean = self.estimated_ctrs[ad_index]
        bonus = self.c * np.sqrt(np.log(t) / self.counts[ad_index])
        return float(empirical_mean + bonus)

    def select_ad(self) -> int:
        unselected = np.where(self.counts == 0)[0]
        if len(unselected) > 0:
            return int(unselected[0])
        t_next = self.current_round + 1
        ucb_values = [self.calculate_ucb(i, t_next) for i in range(self.num_ads)]
        return int(np.argmax(ucb_values))
```

### A.3 Closed-Loop Simulation Harness (`src/simulation.py`)
```python
import numpy as np
import pandas as pd

def run_simulation(env, agent, num_rounds=10000, seed=None):
    env.reset(seed=seed)
    agent.reset()
    true_ctrs = env.get_true_ctrs()
    best_idx, _, max_ctr = env.get_best_ad()

    selected_ads = np.zeros(num_rounds, dtype=np.int64)
    rewards = np.zeros(num_rounds, dtype=np.int64)
    cum_rewards = np.zeros(num_rounds, dtype=np.int64)
    cum_expected_regrets = np.zeros(num_rounds, dtype=np.float64)

    running_reward = 0
    running_regret = 0.0

    for step in range(num_rounds):
        chosen_ad = agent.select_ad()
        reward = env.pull_ad(chosen_ad)
        agent.update(chosen_ad, reward)

        running_reward += reward
        running_regret += (max_ctr - true_ctrs[chosen_ad])

        selected_ads[step] = chosen_ad
        rewards[step] = reward
        cum_rewards[step] = running_reward
        cum_expected_regrets[step] = running_regret

    return pd.DataFrame({
        "round": np.arange(1, num_rounds + 1),
        "selected_ad": selected_ads,
        "reward": rewards,
        "cumulative_reward": cum_rewards,
        "cumulative_expected_regret": cum_expected_regrets,
    })
```

<div style="page-break-after: always;"></div>

---

# APPENDIX B — DETAILED EXPERIMENTAL RESULT TABLES

**Table B.1: Multi-Algorithm Arm Allocation Percentage Breakdown ($T = 10,000$ Rounds)**

| Algorithm Policy | Ad 1 (5%) | Ad 2 (10%) | Ad 3 (3%) | Ad 4 (15%, Best) | Ad 5 (8%) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **$\varepsilon$-Greedy (decay)** | 0.28% | 0.42% | 0.21% | **98.81%** | 0.28% |
| **Thompson Sampling** | 0.65% | 1.84% | 0.32% | **96.42%** | 0.77% |
| **$\varepsilon$-Greedy ($\varepsilon = 0.1$)** | 2.12% | 2.25% | 1.98% | **91.60%** | 2.05% |
| **UCB1 ($c = 2.0$)** | 10.57% | 17.70% | 8.59% | **51.56%** | 11.58% |
| **Random / Static A/B** | 20.00% | 20.00% | 20.00% | **20.00%** | 20.00% |

**Table B.2: Offline Policy Replay Sampling Efficiency (10,000 Logged Impressions)**

| Evaluated Policy | Stream Samples | Accepted Impressions ($T_{\text{acc}}$) | Acceptance Rate | Effective Policy CTR |
|:---|:---:|:---:|:---:|:---:|
| **UCB1 ($c = 1.5$)** | 10,000 | 2,506 | 25.06% | 39.39% |
| **Thompson Sampling** | 10,000 | 2,468 | 24.68% | 42.59% |
| **Contextual LinUCB** | 10,000 | 2,502 | 25.02% | 47.92% |

<div style="page-break-after: always;"></div>

---

# APPENDIX C — DASHBOARD USER INTERFACE VERIFICATION

The actual captured user interface screenshots from the live application verification are presented below with complete academic descriptions.

### C.1 Initial Empty State
![Figure C.1: Dashboard Initial State (Empty Placeholder)](reports/screenshots/01_tab1_initial_empty.png)  
*Figure C.1: Dashboard Initial State. Demonstrates the clean academic workspace prior to execution. Notice the structured configuration card, hidden ground truth CTR inputs, and the explicit action button requirement.*

### C.2 Single-Agent Executed Simulation Results
![Figure C.2: Single-Agent UCB1 Executed Simulation Results](reports/screenshots/02_tab1_executed_results.png)  
*Figure C.2: Single-Agent UCB1 Executed Results. Displays summary KPI callout cards, empirical CTR matrix table, and four publication-quality analytical plots following explicit execution of 10,000 rounds.*

### C.3 Multi-Algorithm Benchmark Arena
![Figure C.3: Multi-Algorithm Benchmark Arena Tab](reports/screenshots/03_tab2_arena.png)  
*Figure C.3: Algorithm Benchmark Arena. Displays comparative tournament leaderboard ranking UCB1, Thompson Sampling, $\varepsilon$-Greedy, and Random policies alongside cumulative reward and regret curves.*

### C.4 Contextual Personalization & Interactive Persona Simulator
![Figure C.4: Contextual Personalization & Interactive Persona Simulator](reports/screenshots/04_tab3_contextual.png)  
*Figure C.4: Contextual Personalization Interface. Features the formal visitor attribute vector $\mathbf{x} \in \mathbb{R}^5$ selector, real-time LinUCB recommendation preview, and personalized CTR simulation curves.*

### C.5 Non-Stationary Dynamics & Ad Fatigue Adaptation
![Figure C.5: Non-Stationary Dynamics & Ad Fatigue Tab](reports/screenshots/05_tab4_non_stationary.png)  
*Figure C.5: Non-Stationary Shift Adaptation. Displays the trajectory following an abrupt mid-stream CTR shift at round 5,000, contrasting Sliding-Window UCB's rapid adaptation against stationary UCB1.*

### C.6 Unbiased Offline Policy Evaluation
![Figure C.6: Unbiased Offline Policy Evaluation Tab](reports/screenshots/06_tab5_offline.png)  
*Figure C.6: Offline Policy Evaluation. Shows the Li et al. (WWW 2010) rejection sampling replay evaluation table across 10,000 logged historical impression events.*

### C.7 Academic Viva Preparation & Methodology Guide
![Figure C.7: Methodology & Academic Viva Preparation Guide](reports/screenshots/07_tab6_methodology.png)  
*Figure C.7: Academic Viva Preparation Guide. Displays high-contrast academic Q&A cards covering algorithmic proofs, exploration bounds, and theoretical foundations.*

### C.8 Responsive Mobile Layout Verification
![Figure C.8: Responsive Mobile Viewport Layout Verification](reports/screenshots/08_mobile_responsive.png)  
*Figure C.8: Responsive Viewport Verification (375px). Confirms that configuration cards, metric blocks, and tables stack gracefully on mobile screen sizes.*
