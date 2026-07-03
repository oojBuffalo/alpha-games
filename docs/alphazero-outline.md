# AlphaZero — Structured Outline
 
**Citation:** Silver, Hubert, Schrittwieser et al. (DeepMind/UCL),
*"A general reinforcement learning algorithm that masters chess, shogi and Go through self-play."*
 
---
 
## 1. Problem & Motivation
 
- Computer chess (Deep Blue, Stockfish) and shogi (Elmo) reached superhuman play via **alpha-beta search + handcrafted evaluation functions** refined by human experts over decades.
- These systems are **domain-tuned** and don't generalize; general game-playing systems were comparatively weak.
- Goal: a **single algorithm** that learns *tabula rasa* (random init, rules only — no human data/knowledge) and reaches superhuman play across multiple games.
---
 
## 2. Core Contribution
 
- Generalizes **AlphaGo Zero** into **AlphaZero**, applied unchanged to chess, shogi, and Go (same algorithm, network architecture, and hyperparameters).
- Replaces three traditional components with general-purpose ones:
| Traditional | AlphaZero |
|---|---|
| Handcrafted evaluation function | Deep neural network |
| Alpha-beta search | Monte Carlo Tree Search (MCTS) |
| Human-tuned weights | Reinforcement learning from self-play |
 
---
 
## 3. Method
 
### 3.1 Neural Network
 
- Single network $(p, v) = f_\theta(s)$:
  - $p$ — move-probability vector, $p_a = \Pr(a \mid s)$
  - $v$ — scalar value estimating expected game outcome, $v \approx \mathbb{E}[z \mid s]$
- Convolutional (ResNet) architecture, identical across all three games.
### 3.2 Search (MCTS)
 
- Each search runs many simulated self-play games from root $s_\text{root}$ to a leaf.
- Move selection at each node balances:
  - **Low visit count** (exploration)
  - **High prior probability** $p_a$
  - **High mean value** (averaged over simulations)
- Returns visit-count distribution $\pi$ over root moves; actual move sampled proportionally (explore) or greedily (exploit).
- Highly selective: ~60,000 positions/sec vs. Stockfish's 60M and Elmo's 25M — a more "human-like" search (per Shannon).
### 3.3 Training Loss
 
Parameters updated by gradient descent minimizing:
 
$$l = (z - v)^2 - \pi^\top \log p + c\lVert\theta\rVert^2$$
 
- $z \in \{-1, 0, +1\}$ — terminal outcome (loss/draw/win)
- First term: MSE on value head
- Second term: cross-entropy on policy head
- Third term: L2 weight regularization
---
 
## 4. Differences from AlphaGo Zero
 
| Feature | AlphaGo Zero | AlphaZero |
|---|---|---|
| Outcome modeling | Win probability (binary) | Expected outcome (handles draws) |
| Symmetry augmentation | 8-fold (Go is symmetric) | None (chess/shogi are asymmetric) |
| Network update rule | Best-player gating (55% threshold) | Continuous single-network updates |
| Hyperparameter tuning | Bayesian optimization per game | Reused across all games |
 
- Only game-specific adjustments: **exploration noise** and **learning-rate schedule**.
---
 
## 5. Training Setup
 
- **Steps:** 700,000 mini-batches of 4,096 positions from random initialization.
- **Hardware:**
  - 5,000 first-gen TPUs — self-play game generation
  - 16 second-gen TPUs — neural network training
- **Wall-clock training time:**
  - Chess: ~9 hours
  - Shogi: ~12 hours
  - Go: ~13 days
- **Time to surpass prior SOTA:**
  - Stockfish: 4 hours (300,000 steps)
  - Elmo: 2 hours (110,000 steps)
  - AlphaGo Lee: 30 hours (74,000 steps)
---
 
## 6. Evaluation & Results
 
**Match conditions:** 3 h/game + 15 s/move increment; AlphaZero on 4 first-gen TPUs + 44 CPU cores; opponents on 44 CPU cores.
 
### 6.1 Chess vs. Stockfish (1,000 games)
 
- **155 wins, 6 losses**, rest draws.
- Won from human openings, TCEC championship openings, newest Stockfish build, and Stockfish with an opening book.
- Qualitative: sacrifices pieces for long-term positional advantage; independently rediscovered common human openings.
### 6.2 Shogi vs. Elmo
 
- **98.2% win rate as Black, 91.2% overall.**
- Also won under faster CSA championship time controls and vs. Aperyqhapaq (another SOTA program).
### 6.3 Go vs. AlphaGo Zero
 
- **61% win rate** — recovers full performance despite forgoing symmetry-based 8× data augmentation.
### 6.4 Time Handicap Tests
 
- Beat Stockfish at **1/10 thinking time** (~1/10,000 positions searched).
- Won **46% vs. Elmo at 1/100 thinking time** (~1/40,000 positions searched).
---
 
## 7. Significance
 
- Demonstrates a **single general-purpose RL + search algorithm** mastering three distinct games with no domain-specific engineering.
- Challenges the long-held belief that **alpha-beta search is inherently superior** to MCTS in combinatorial game domains.
- A concrete step toward a general game-playing system that can learn to master any game from rules alone.
