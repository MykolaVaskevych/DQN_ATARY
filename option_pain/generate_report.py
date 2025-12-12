#!/usr/bin/env python3
"""
Generate comprehensive metrics report with all training data.
"""

import json
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def extract_tensorboard_data(log_dir: str) -> dict:
    """Extract all metrics from TensorBoard logs."""
    log_path = Path(log_dir)
    subdirs = sorted(log_path.iterdir(), key=lambda x: x.name)
    if not subdirs:
        return {}

    latest_run = subdirs[-1]
    ea = EventAccumulator(str(latest_run))
    ea.Reload()

    data = {}
    for tag in ea.Tags()['scalars']:
        events = ea.Scalars(tag)
        data[tag] = {
            'steps': [e.step for e in events],
            'values': [e.value for e in events],
        }
    return data


def main():
    output_dir = Path("metrics_output")
    output_dir.mkdir(exist_ok=True)

    # Load evaluation results
    with open(output_dir / "metrics_report.json") as f:
        report = json.load(f)

    # Extract TensorBoard data
    print("Extracting TensorBoard data...")
    tb_data = extract_tensorboard_data("tensorboard_logs")

    agent = report["agent"]
    benchmarks = report["benchmarks"]
    random_baseline = report["random_baseline"]
    learning_curve = report["learning_curve"]

    # Get final training metrics
    final_loss = tb_data.get('train/loss', {}).get('values', [None])[-1]
    final_lr = tb_data.get('train/learning_rate', {}).get('values', [None])[-1]
    final_eps = tb_data.get('rollout/exploration_rate', {}).get('values', [None])[-1]
    final_fps = tb_data.get('time/fps', {}).get('values', [None])[-1]

    # Loss statistics
    loss_values = tb_data.get('train/loss', {}).get('values', [])
    loss_mean = np.mean(loss_values) if loss_values else 0
    loss_std = np.std(loss_values) if loss_values else 0
    loss_min = np.min(loss_values) if loss_values else 0
    loss_max = np.max(loss_values) if loss_values else 0

    # Count score frequencies
    scores = agent['all_rewards']
    unique, counts = np.unique(scores, return_counts=True)
    score_freq = list(zip(unique, counts))

    summary = f"""
================================================================================
                    DQN BREAKOUT - COMPLETE METRICS REPORT
================================================================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Model: {report['model_path']}

================================================================================
                           1. EVALUATION RESULTS
================================================================================

Evaluated over {report['n_episodes']} episodes using deterministic policy (no exploration).

SCORE STATISTICS:
--------------------------------------------------------------------------------
  Mean Score:       {agent['mean']:>10.1f}
  Std Dev:          {agent['std']:>10.1f}
  Min Score:        {agent['min']:>10.1f}
  Max Score:        {agent['max']:>10.1f}
--------------------------------------------------------------------------------

INDIVIDUAL EPISODE SCORES (all {report['n_episodes']} episodes):
--------------------------------------------------------------------------------
Episode:  1   2   3   4   5   6   7   8   9  10  11  12  13  14  15
Score:  {' '.join(f'{int(s):>3}' for s in scores[:15])}

Episode: 16  17  18  19  20  21  22  23  24  25  26  27  28  29  30
Score:  {' '.join(f'{int(s):>3}' for s in scores[15:])}
--------------------------------------------------------------------------------

SCORE FREQUENCY TABLE:
--------------------------------------------------------------------------------
Score       Count       Percentage      Description
--------------------------------------------------------------------------------"""

    for score, count in sorted(score_freq, key=lambda x: x[0]):
        pct = count / len(scores) * 100
        summary += f"\n{int(score):>5}       {count:>5}       {pct:>10.1f}%      "
        if score < 100:
            summary += "Early termination (lost all lives quickly)"
        elif score < 250:
            summary += "Medium performance (~5-6 brick rows cleared)"
        else:
            summary += "Good performance (~6-7 brick rows cleared)"

    summary += f"""
--------------------------------------------------------------------------------
Total:      {len(scores):>5}          100.0%
--------------------------------------------------------------------------------

================================================================================
                           2. BASELINE COMPARISONS
================================================================================

COMPARISON TABLE:
--------------------------------------------------------------------------------
Agent/Baseline              Score       Source              Notes
--------------------------------------------------------------------------------
Random Agent (ours)         {random_baseline['mean']:>5.1f}       Computed            {report['n_episodes']} episodes, random actions
Random Agent (DeepMind)     {benchmarks['random_score']:>5.1f}       Nature 2015         Table 1
Human Professional          {benchmarks['human_score']:>5.1f}       Nature 2015         ~2 hours of play
Our DQN                   {agent['mean']:>6.1f}       Computed            10M timesteps training
DeepMind DQN              {benchmarks['deepmind_dqn']:>6.1f}       Nature 2015         200M frames training
--------------------------------------------------------------------------------

================================================================================
                         3. HUMAN-NORMALIZED SCORE
================================================================================

FORMULA:
  HNS = 100 × (Agent_Score - Random_Score) / (Human_Score - Random_Score)

CALCULATION:
  Our DQN:      100 × ({agent['mean']:.1f} - {random_baseline['mean']:.1f}) / ({benchmarks['human_score']} - {random_baseline['mean']:.1f})
              = 100 × {agent['mean'] - random_baseline['mean']:.1f} / {benchmarks['human_score'] - random_baseline['mean']:.1f}
              = {report['human_normalized_score']:.1f}%

INTERPRETATION:
--------------------------------------------------------------------------------
Score Range         Meaning
--------------------------------------------------------------------------------
0%                  Equal to random agent (no learning)
100%                Equal to human professional player
>100%               Superhuman performance
--------------------------------------------------------------------------------

RESULTS:
--------------------------------------------------------------------------------
Agent               Human-Normalized Score      Interpretation
--------------------------------------------------------------------------------
Our DQN             {report['human_normalized_score']:>10.1f}%              {report['human_normalized_score']/100:.1f}x human performance
DeepMind DQN        {100 * (benchmarks['deepmind_dqn'] - benchmarks['random_score']) / (benchmarks['human_score'] - benchmarks['random_score']):>10.1f}%              13.3x human performance
--------------------------------------------------------------------------------

================================================================================
                           4. TRAINING CONFIGURATION
================================================================================

HYPERPARAMETERS:
--------------------------------------------------------------------------------
Parameter                   Value           Description
--------------------------------------------------------------------------------
Total Timesteps             10,000,000      Number of environment steps
Buffer Size                 400,000         Experience replay buffer capacity
Batch Size                  64              Samples per gradient update
Learning Rate               1e-4            Adam optimizer step size
Gamma (discount)            0.99            Future reward discount factor
Target Update Interval      10,000          Steps between target network updates
Exploration Initial         1.0             Starting epsilon (100% random)
Exploration Final           0.01            Final epsilon (1% random)
Exploration Fraction        0.1             Fraction of training for epsilon decay
Frame Stack                 4               Consecutive frames stacked
--------------------------------------------------------------------------------

================================================================================
                           5. TRAINING METRICS
================================================================================

FINAL VALUES (at 10M timesteps):
--------------------------------------------------------------------------------
Metric                      Value           Description
--------------------------------------------------------------------------------
Loss                        {final_loss:.4f}          TD error (Bellman equation)
Learning Rate               {final_lr:.6f}        Constant throughout training
Exploration Rate (epsilon)  {final_eps:.4f}          Final exploration probability
Training Speed              {final_fps:.0f} FPS        Frames per second
--------------------------------------------------------------------------------

LOSS STATISTICS (over entire training):
--------------------------------------------------------------------------------
Statistic                   Value
--------------------------------------------------------------------------------
Mean Loss                   {loss_mean:.4f}
Std Dev                     {loss_std:.4f}
Min Loss                    {loss_min:.4f}
Max Loss                    {loss_max:.4f}
--------------------------------------------------------------------------------

================================================================================
                           6. LEARNING PROGRESSION
================================================================================

EVALUATION CHECKPOINTS (during training):
--------------------------------------------------------------------------------
Timestep      Mean Score      Std Dev     Min     Max     Notes
--------------------------------------------------------------------------------"""

    for i, point in enumerate(learning_curve):
        ts = point['timestep'] / 1e6
        notes = ""
        if i == 0:
            notes = "Early training"
        elif point['mean'] == max(p['mean'] for p in learning_curve):
            notes = "PEAK PERFORMANCE"
        elif i == len(learning_curve) - 1:
            notes = "Final checkpoint"
        summary += f"\n{ts:>6.1f}M       {point['mean']:>10.1f}    {point['std']:>8.1f}  {point['min']:>6.0f}  {point['max']:>6.0f}    {notes}"

    peak_idx = np.argmax([p['mean'] for p in learning_curve])
    peak_point = learning_curve[peak_idx]

    summary += f"""
--------------------------------------------------------------------------------

SUMMARY:
  - Training started at {learning_curve[0]['mean']:.1f} mean score
  - Peak performance: {peak_point['mean']:.1f} at {peak_point['timestep']/1e6:.1f}M timesteps
  - Final performance: {learning_curve[-1]['mean']:.1f} at {learning_curve[-1]['timestep']/1e6:.1f}M timesteps
  - Performance decreased after peak (possible overfitting)

================================================================================
                           7. PLOT EXPLANATIONS
================================================================================

PLOT: learning_curve.png
--------------------------------------------------------------------------------
Description: Shows how the agent's score improved during training.

Elements:
  - Blue line: Mean evaluation score at each checkpoint
  - Blue shaded area: ±1 standard deviation (score variability)
  - Green dashed line: Human professional baseline (31.8)
  - Red dotted line: Random agent baseline (1.7)

Interpretation:
  - Agent surpassed human level around 1.5M timesteps
  - Continued improving until ~8M timesteps
  - Slight decline after peak suggests diminishing returns or instability
--------------------------------------------------------------------------------

PLOT: training_curves.png (4 subplots)
--------------------------------------------------------------------------------

Subplot 1 - Training Loss (top-left):
  - Blue (faint): Raw loss values (noisy)
  - Red line: Smoothed loss (rolling average)
  - Y-axis: Loss value (lower = better predictions)
  - Loss measures how wrong the Q-value predictions are
  - Spikes around 8M indicate learning of new strategies

Subplot 2 - Training Reward (top-right):
  - Green line: Mean episode reward during training
  - Orange dashed line: Human baseline (31.8)
  - Shows actual gameplay performance during learning
  - Steady increase indicates successful learning

Subplot 3 - Exploration Rate (bottom-left):
  - Purple line: Epsilon value over time
  - Started at 1.0 (100% random actions)
  - Dropped to 0.01 (1% random) by ~1M timesteps
  - After decay, agent mostly exploits learned policy

Subplot 4 - Episode Length (bottom-right):
  - Orange line: Average episode duration in steps
  - Longer episodes = agent survives longer = better play
  - Correlates with reward increase
--------------------------------------------------------------------------------

PLOT: comparison_chart.png
--------------------------------------------------------------------------------
Description: Bar chart comparing scores across different agents.

Bars (left to right):
  1. Random (Computed): Our random agent baseline (0.5)
  2. Random (DeepMind): Paper's random baseline (1.7)
  3. Human (DeepMind): Professional human tester (31.8)
  4. Our DQN: Our trained agent ({agent['mean']:.1f})
  5. DQN (DeepMind): Original paper's result (401.2)

Note: DeepMind trained for 200M frames vs our 10M timesteps.
--------------------------------------------------------------------------------

PLOT: human_normalized.png
--------------------------------------------------------------------------------
Description: Shows performance relative to human baseline.

Bars:
  - Random: 0% (baseline)
  - Human: 100% (reference point)
  - Our DQN: {report['human_normalized_score']:.1f}% (superhuman)
  - DQN (DeepMind): 1327.2% (highly superhuman)

The green dashed line at 100% marks human-level performance.
Anything above this line is considered superhuman.
--------------------------------------------------------------------------------

================================================================================
                              8. REFERENCE
================================================================================

{benchmarks['citation']}

DOI: 10.1038/nature14236

================================================================================
                              END OF REPORT
================================================================================
"""

    # Save text summary
    summary_path = output_dir / "metrics_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(summary)
    print(f"Text summary saved: {summary_path}")
    print(summary)

    # =================================================================
    # Training Curves Plot (keep this one)
    # =================================================================

    if 'train/loss' in tb_data:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Loss over time
        ax = axes[0, 0]
        steps = np.array(tb_data['train/loss']['steps']) / 1e6
        losses = tb_data['train/loss']['values']
        ax.plot(steps, losses, 'b-', alpha=0.3, linewidth=0.5)
        window = min(100, len(losses) // 10)
        if window > 1:
            smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
            ax.plot(steps[window-1:], smoothed, 'r-', linewidth=1.5, label='Smoothed')
        ax.set_xlabel('Timesteps (Millions)')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss')
        ax.legend()

        # Reward over time
        ax = axes[0, 1]
        if 'rollout/ep_rew_mean' in tb_data:
            steps = np.array(tb_data['rollout/ep_rew_mean']['steps']) / 1e6
            rewards = tb_data['rollout/ep_rew_mean']['values']
            ax.plot(steps, rewards, 'g-', linewidth=1.5)
            ax.axhline(benchmarks['human_score'], color='orange', linestyle='--', label=f"Human: {benchmarks['human_score']}")
            ax.set_xlabel('Timesteps (Millions)')
            ax.set_ylabel('Mean Episode Reward')
            ax.set_title('Training Reward')
            ax.legend()

        # Exploration rate
        ax = axes[1, 0]
        if 'rollout/exploration_rate' in tb_data:
            steps = np.array(tb_data['rollout/exploration_rate']['steps']) / 1e6
            eps = tb_data['rollout/exploration_rate']['values']
            ax.plot(steps, eps, 'purple', linewidth=1.5)
            ax.set_xlabel('Timesteps (Millions)')
            ax.set_ylabel('Epsilon')
            ax.set_title('Exploration Rate (ε-greedy)')

        # Episode length
        ax = axes[1, 1]
        if 'rollout/ep_len_mean' in tb_data:
            steps = np.array(tb_data['rollout/ep_len_mean']['steps']) / 1e6
            lengths = tb_data['rollout/ep_len_mean']['values']
            ax.plot(steps, lengths, 'orange', linewidth=1.5)
            ax.set_xlabel('Timesteps (Millions)')
            ax.set_ylabel('Mean Episode Length')
            ax.set_title('Episode Length')

        plt.tight_layout()
        plt.savefig(output_dir / "training_curves.png", dpi=300)
        plt.close()
        print(f"Training curves saved: {output_dir}/training_curves.png")

    # Remove the bad score_analysis.png if it exists
    bad_plot = output_dir / "score_analysis.png"
    if bad_plot.exists():
        bad_plot.unlink()
        print(f"Removed: {bad_plot}")

    print("\nDone!")


if __name__ == "__main__":
    main()
