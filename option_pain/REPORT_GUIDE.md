# DQN Atari Breakout - Project Report Guide

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [How DQN Works](#2-how-dqn-works)
3. [Implementation Details](#3-implementation-details)
4. [Training Process](#4-training-process)
5. [Results & Evaluation](#5-results--evaluation)
6. [How to Generate Plots](#6-how-to-generate-plots)
7. [Evaluation Metrics Explained](#7-evaluation-metrics-explained)
8. [Commands Reference](#8-commands-reference)

---

## 1. Project Overview

### What We Built
A Deep Q-Network (DQN) agent that learns to play Atari Breakout through reinforcement learning. The agent observes game frames (pixels) and learns which actions (left, right, fire, noop) maximize its score.

### Project Timeline
| Step | What We Did | Result |
|------|-------------|--------|
| 1 | Set up environment with UV, Gymnasium, Stable-Baselines3 | Working Python environment |
| 2 | Created training script with DQN | `train.py` |
| 3 | Created evaluation script with rendering | `evaluate.py` |
| 4 | Trained for 1M steps (test run) | ~21 avg reward |
| 5 | Optimized hyperparameters for RTX 4090 + 32GB RAM | 80% resource utilization |
| 6 | Trained for 10M steps (overnight) | **222 avg reward** (peak) |

### Files Structure
```
atary/
├── train.py              # Training script
├── evaluate.py           # Evaluation/visualization script
├── best_model/           # Best performing model (222 reward)
├── checkpoints/          # Training checkpoints
├── logs/                 # Evaluation logs (for plots)
├── models_1m/            # Backup of 1M training run
└── tensorboard_logs/     # TensorBoard training logs
```

---

## 2. How DQN Works

### The Problem: Learning from Pixels
The agent sees raw game frames (210x160 RGB pixels) and must learn to play without any game-specific programming. It only receives:
- **State**: Game screen pixels
- **Actions**: 4 choices (NOOP, FIRE, LEFT, RIGHT)
- **Reward**: +1 for each brick broken, 0 otherwise

### The Solution: Deep Q-Network

#### Q-Learning Basics
Q-Learning learns a function Q(s, a) that estimates the expected future reward for taking action `a` in state `s`.

```
Q(state, action) = Expected total future reward
```

The agent picks actions with highest Q-value:
```
best_action = argmax(Q(state, all_actions))
```

#### Why "Deep"?
Traditional Q-learning uses a table to store Q-values. With images (84×84×4 = 28,224 dimensions), a table is impossible. Instead, we use a **Convolutional Neural Network (CNN)** to approximate Q-values.

```
Input: 84x84x4 grayscale frames
  ↓
Conv Layer 1: 32 filters, 8x8, stride 4
  ↓
Conv Layer 2: 64 filters, 4x4, stride 2
  ↓
Conv Layer 3: 64 filters, 3x3, stride 1
  ↓
Fully Connected: 512 units
  ↓
Output: 4 Q-values (one per action)
```

### Key DQN Innovations

#### 1. Experience Replay
Instead of learning from consecutive frames (which are correlated), DQN stores experiences in a **replay buffer** and samples random mini-batches.

```
Replay Buffer (400,000 transitions):
┌─────────────────────────────────────────┐
│ (state₁, action₁, reward₁, next_state₁) │
│ (state₂, action₂, reward₂, next_state₂) │
│ ...                                      │
│ (stateₙ, actionₙ, rewardₙ, next_stateₙ) │
└─────────────────────────────────────────┘
         ↓ Random sample
   Mini-batch of 64 transitions
         ↓
   Gradient update
```

**Why it helps**: Breaks correlation between consecutive samples, allows reuse of rare experiences.

#### 2. Target Network
DQN uses two networks:
- **Online Network**: Updated every step
- **Target Network**: Frozen copy, updated every 10,000 steps

```
Q_target = reward + γ × max(Q_target_network(next_state))
Loss = (Q_online(state, action) - Q_target)²
```

**Why it helps**: Stabilizes training by keeping targets fixed.

#### 3. Frame Stacking
A single frame doesn't show motion (ball direction, paddle velocity). We stack 4 consecutive frames:

```
Frame t-3  Frame t-2  Frame t-1  Frame t
   ↓          ↓          ↓         ↓
┌──────┬──────┬──────┬──────┐
│      │      │      │      │  → 84x84x4 input
│  ○   │   ○  │    ○ │     ○│    (ball moving right)
└──────┴──────┴──────┴──────┘
```

#### 4. Epsilon-Greedy Exploration
Balance between exploring new actions and exploiting known good ones:

```
if random() < epsilon:
    action = random_action()      # Explore
else:
    action = argmax(Q(state))     # Exploit

epsilon: 1.0 → 0.01 over first 1M steps
```

### Training Loop Pseudocode
```python
for step in range(10_000_000):
    # 1. Select action (epsilon-greedy)
    if random() < epsilon:
        action = random_action()
    else:
        action = argmax(Q_network(state))

    # 2. Execute action, observe result
    next_state, reward, done = env.step(action)

    # 3. Store in replay buffer
    buffer.add(state, action, reward, next_state, done)

    # 4. Sample mini-batch and train (every 4 steps)
    if step % 4 == 0:
        batch = buffer.sample(64)

        # Compute target Q-values
        target = reward + 0.99 * max(Q_target(next_state))

        # Update online network
        loss = (Q_online(state, action) - target)²
        optimizer.step(loss)

    # 5. Update target network (every 10,000 steps)
    if step % 10000 == 0:
        Q_target = copy(Q_online)
```

---

## 3. Implementation Details

### Environment Preprocessing
Raw Atari frames undergo several transformations:

| Step | Input | Output | Why |
|------|-------|--------|-----|
| Frame Skip | 1 frame | 4 frames | Reduce computation, add temporal info |
| Max Pool | 2 frames | 1 frame | Reduce flickering |
| Grayscale | RGB (3 channels) | Gray (1 channel) | Reduce dimensions |
| Resize | 210×160 | 84×84 | Standard input size |
| Frame Stack | 1 frame | 4 frames | Capture motion |

### Hyperparameters Used

| Parameter | Value | Explanation |
|-----------|-------|-------------|
| `learning_rate` | 0.0001 | Step size for gradient descent |
| `buffer_size` | 400,000 | Number of transitions stored |
| `batch_size` | 64 | Samples per gradient update |
| `gamma` | 0.99 | Discount factor for future rewards |
| `exploration_fraction` | 0.1 | Fraction of training for epsilon decay |
| `exploration_initial_eps` | 1.0 | Starting exploration rate |
| `exploration_final_eps` | 0.01 | Final exploration rate |
| `target_update_interval` | 10,000 | Steps between target network updates |
| `learning_starts` | 50,000 | Steps before training begins |
| `train_freq` | 4 | Environment steps between updates |
| `n_envs` | 8 | Parallel environments |

### Hardware Configuration
- **GPU**: NVIDIA RTX 4090 (24GB VRAM)
- **RAM**: 32GB
- **Resource Usage**: ~80% of available memory

---

## 4. Training Process

### Training Progression (10M Steps)

| Timesteps | Avg Reward | Episode Length | Notes |
|-----------|------------|----------------|-------|
| 800K | 18.6 | 2,181 | Learning basic paddle control |
| 1.6M | 44.2 | 4,530 | Improved ball tracking |
| 2.4M | 54.2 | 5,266 | Consistent gameplay |
| 3.2M | 63.6 | 5,166 | Better brick targeting |
| 4.0M | 70.0 | 5,058 | Refined strategy |
| 4.8M | 80.4 | 5,753 | Longer rallies |
| 5.6M | 95.8 | 5,960 | Breaking more bricks |
| 6.4M | 129.4 | 6,053 | Significant improvement |
| 7.2M | 174.6 | 6,090 | Near-peak performance |
| **8.0M** | **222.4** | **6,804** | **Peak performance** |
| 8.8M | 206.6 | 5,648 | Slight decline |
| 9.6M | 194.0 | 5,707 | Stabilized |

### Key Observations
1. **Rapid early learning**: 0→70 reward in first 4M steps
2. **Continued improvement**: 70→222 reward from 4M to 8M
3. **Peak at 8M**: Best model saved automatically
4. **Slight decline after peak**: Normal variance, best model preserved

---

## 5. Results & Evaluation

### Final Results Summary

| Metric | Value |
|--------|-------|
| Best Training Reward | 222.4 |
| Final Training Reward | 194.0 |
| Peak Episode Length | 6,804 steps |
| Training Time | ~2 hours |
| Total Timesteps | 10,000,000 |

### Comparison with Baselines

| Agent | Breakout Score | Notes |
|-------|----------------|-------|
| Random Agent | ~1-2 | No learning |
| **Our DQN (1M steps)** | **21.6** | Initial training |
| **Our DQN (10M steps)** | **222.4** | Full training |
| Original DQN Paper (2015) | ~401 | 50M frames |
| Human Performance | ~31-80 | Varies by source |
| State-of-the-art | 500+ | Advanced variants |

### Improvement Analysis
- **1M → 10M training**: 10x improvement (21.6 → 222.4)
- **vs Random**: 100x+ improvement
- **vs Human**: 3-7x better than average human

---

## 6. How to Generate Plots

### Plot 1: Training Reward Curve
```bash
uv run python -c "
import numpy as np
import matplotlib.pyplot as plt

# Load data
data = np.load('logs/evaluations.npz')
timesteps = data['timesteps'] / 1e6  # Convert to millions
rewards = data['results'].mean(axis=1)
std = data['results'].std(axis=1)

# Create plot
plt.figure(figsize=(10, 6))
plt.plot(timesteps, rewards, 'b-', linewidth=2, label='Mean Reward')
plt.fill_between(timesteps, rewards - std, rewards + std, alpha=0.3, label='±1 Std Dev')
plt.xlabel('Training Steps (Millions)', fontsize=12)
plt.ylabel('Average Reward', fontsize=12)
plt.title('DQN Training Progress on Atari Breakout', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('reward_curve.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: reward_curve.png')
"
```

### Plot 2: Episode Length Over Training
```bash
uv run python -c "
import numpy as np
import matplotlib.pyplot as plt

data = np.load('logs/evaluations.npz')
timesteps = data['timesteps'] / 1e6
lengths = data['ep_lengths'].mean(axis=1)

plt.figure(figsize=(10, 6))
plt.plot(timesteps, lengths, 'g-', linewidth=2)
plt.xlabel('Training Steps (Millions)', fontsize=12)
plt.ylabel('Average Episode Length (steps)', fontsize=12)
plt.title('Episode Duration Over Training', fontsize=14)
plt.grid(True, alpha=0.3)
plt.savefig('episode_length.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: episode_length.png')
"
```

### Plot 3: Reward Distribution (Box Plot)
```bash
uv run python -c "
import numpy as np
import matplotlib.pyplot as plt

data = np.load('logs/evaluations.npz')
timesteps = data['timesteps'] / 1e6
results = data['results']  # Shape: (n_evals, n_episodes)

# Select key checkpoints
indices = [0, 3, 6, 9, 11]  # 0.8M, 3.2M, 6.4M, 8.8M, 9.6M
labels = [f'{timesteps[i]:.1f}M' for i in indices]
box_data = [results[i] for i in indices]

plt.figure(figsize=(10, 6))
plt.boxplot(box_data, labels=labels)
plt.xlabel('Training Steps', fontsize=12)
plt.ylabel('Reward', fontsize=12)
plt.title('Reward Distribution at Different Training Stages', fontsize=14)
plt.grid(True, alpha=0.3, axis='y')
plt.savefig('reward_boxplot.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: reward_boxplot.png')
"
```

### Plot 4: Comparison Bar Chart (1M vs 10M)
```bash
uv run python -c "
import matplotlib.pyplot as plt
import numpy as np

models = ['Random', 'DQN 1M Steps', 'DQN 10M Steps', 'Human (avg)']
scores = [1.5, 21.6, 222.4, 50]
colors = ['gray', 'steelblue', 'darkblue', 'green']

plt.figure(figsize=(10, 6))
bars = plt.bar(models, scores, color=colors)
plt.ylabel('Average Score', fontsize=12)
plt.title('Performance Comparison: DQN vs Baselines', fontsize=14)
plt.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, score in zip(bars, scores):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
             f'{score:.1f}', ha='center', fontsize=11)

plt.savefig('comparison_bar.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: comparison_bar.png')
"
```

### TensorBoard (Interactive Plots)
```bash
# Launch TensorBoard for detailed training metrics
uv run tensorboard --logdir tensorboard_logs/

# Open browser to http://localhost:6006
# Available metrics:
# - rollout/ep_len_mean
# - rollout/ep_rew_mean
# - rollout/exploration_rate
# - train/loss
# - train/learning_rate
```

---

## 7. Evaluation Metrics Explained

### Metrics Table

| Metric | What It Measures | Good Value | Our Result |
|--------|------------------|------------|------------|
| **Mean Reward** | Average score per episode | Higher = better | 222.4 |
| **Std Deviation** | Consistency of performance | Lower = more consistent | ~50 |
| **Episode Length** | How long agent survives | Longer = better | 6,804 steps |
| **Max Reward** | Best single episode | Shows potential | 300+ |
| **Min Reward** | Worst single episode | Shows failure cases | ~50 |

### Evaluation Matrix Template

```
┌─────────────────────────────────────────────────────────────┐
│                    EVALUATION MATRIX                         │
├─────────────────┬───────────┬───────────┬──────────────────┤
│ Metric          │ 1M Steps  │ 10M Steps │ Improvement      │
├─────────────────┼───────────┼───────────┼──────────────────┤
│ Mean Reward     │ 21.6      │ 222.4     │ +930% (10.3x)    │
│ Max Reward      │ 35        │ 300+      │ +757%            │
│ Episode Length  │ 2,494     │ 6,804     │ +173% (2.7x)     │
│ Training Time   │ 12 min    │ 2 hours   │ 10x longer       │
│ Timesteps       │ 1M        │ 10M       │ 10x more         │
└─────────────────┴───────────┴───────────┴──────────────────┘
```

### Generate Evaluation Statistics
```bash
uv run python -c "
import numpy as np

data = np.load('logs/evaluations.npz')
results = data['results']
lengths = data['ep_lengths']

print('='*60)
print('FINAL MODEL EVALUATION STATISTICS (10M Steps)')
print('='*60)
print(f'Mean Reward:        {results[-1].mean():.2f}')
print(f'Std Reward:         {results[-1].std():.2f}')
print(f'Max Reward:         {results[-1].max():.0f}')
print(f'Min Reward:         {results[-1].min():.0f}')
print(f'Mean Episode Len:   {lengths[-1].mean():.0f}')
print()
print('BEST MODEL STATISTICS (8M Steps)')
print('='*60)
best_idx = results.mean(axis=1).argmax()
print(f'Mean Reward:        {results[best_idx].mean():.2f}')
print(f'Std Reward:         {results[best_idx].std():.2f}')
print(f'Max Reward:         {results[best_idx].max():.0f}')
print(f'Min Reward:         {results[best_idx].min():.0f}')
print(f'Mean Episode Len:   {lengths[best_idx].mean():.0f}')
"
```

---

## 8. Commands Reference

### Training
```bash
# Train new model
uv run python train.py

# Train with custom timesteps (edit train.py line 141)
# total_timesteps = 50_000_000  # For overnight training
```

### Evaluation
```bash
# Evaluate best model (5 episodes, rendered)
uv run python evaluate.py --best

# Evaluate with more episodes
uv run python evaluate.py --best --episodes 20

# Evaluate specific checkpoint
uv run python evaluate.py --model checkpoints/dqn_breakout_8000000_steps

# Evaluate with some randomness (stochastic)
uv run python evaluate.py --best --stochastic
```

### Generate All Plots
```bash
# Install matplotlib if needed
uv add matplotlib

# Run all plot commands from Section 6
```

### View Training Logs
```bash
# TensorBoard
uv run tensorboard --logdir tensorboard_logs/

# Raw evaluation data
uv run python -c "
import numpy as np
data = np.load('logs/evaluations.npz')
print('Keys:', list(data.keys()))
print('Timesteps:', data['timesteps'])
print('Results shape:', data['results'].shape)
"
```

---

## Appendix: Report Outline Suggestion

### Suggested Report Structure
1. **Introduction**
   - What is reinforcement learning?
   - What is the Atari Breakout problem?
   - Project objectives

2. **Background**
   - Q-Learning theory
   - Deep Q-Networks (DQN)
   - Key innovations (replay buffer, target network)

3. **Methodology**
   - Environment setup
   - Network architecture
   - Hyperparameters
   - Training procedure

4. **Results**
   - Training curves (plots)
   - Final performance metrics
   - Comparison with baselines

5. **Discussion**
   - What worked well
   - Limitations
   - Possible improvements

6. **Conclusion**
   - Summary of achievements
   - Future work

### Key Figures to Include
1. Training reward curve (with std deviation)
2. Episode length over training
3. Comparison bar chart (random vs DQN vs human)
4. Screenshot of agent playing (from evaluation)
5. Network architecture diagram
6. Reward distribution box plot

---

*Generated for DQN Atari Breakout project - December 2024*
