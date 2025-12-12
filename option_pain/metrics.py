#!/usr/bin/env python3
"""
DQN Breakout Performance Metrics Evaluation Script

Implements DeepMind's evaluation protocol from:
    Mnih, V., et al. (2015). "Human-level control through deep reinforcement
    learning." Nature, 518(7540), 529-533. doi:10.1038/nature14236

Reference values from Table 1 of the paper.
"""

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import ale_py  # Registers Atari ROMs
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# =============================================================================
# REFERENCE VALUES - DeepMind Nature 2015 Paper, Table 1
# DOI: 10.1038/nature14236
# =============================================================================

BENCHMARKS = {
    "random_score": 1.7,      # Random agent (Table 1)
    "human_score": 31.8,      # Professional human tester (Table 1)
    "deepmind_dqn": 401.2,    # DQN after 200M frames (Table 1)
    "citation": (
        "Mnih, V., et al. (2015). Human-level control through deep "
        "reinforcement learning. Nature, 518(7540), 529-533."
    ),
}


def compute_human_normalized_score(agent_score: float, random_score: float, human_score: float) -> float:
    """
    DeepMind's human-normalized score formula.

    HNS = 100 × (agent - random) / (human - random)

    0% = random, 100% = human, >100% = superhuman
    """
    if human_score == random_score:
        return 0.0
    return 100.0 * (agent_score - random_score) / (human_score - random_score)


def load_learning_curve(npz_path: str) -> list[dict]:
    """Load learning curve from SB3's evaluations.npz file."""
    path = Path(npz_path)
    if not path.exists():
        return []

    data = np.load(npz_path)
    points = []
    for i, ts in enumerate(data['timesteps']):
        scores = data['results'][i]
        points.append({
            "timestep": int(ts),
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
        })
    return points


def setup_plots():
    """Configure matplotlib for publication quality."""
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'figure.figsize': (10, 6),
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.grid': True,
        'grid.alpha': 0.3,
    })


def main():
    parser = argparse.ArgumentParser(description="Evaluate DQN on Breakout")
    parser.add_argument('--model', '-m', default='best_model/best_model.zip', help='Model path')
    parser.add_argument('--episodes', '-n', type=int, default=30, help='Evaluation episodes (default: 30)')
    parser.add_argument('--output-dir', '-o', default='metrics_output', help='Output directory')
    parser.add_argument('--deterministic', action='store_true', help='Use deterministic actions (no exploration)')
    parser.add_argument('--skip-random', action='store_true', help='Skip random baseline computation')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print("\n" + "=" * 60)
    print("  DQN BREAKOUT METRICS - DeepMind Protocol")
    print("=" * 60)

    # Load model
    print(f"\nLoading model: {args.model}")
    model = DQN.load(args.model)

    # Create evaluation environment (same as training)
    print("Creating evaluation environment...")
    eval_env = make_atari_env("BreakoutNoFrameskip-v4", n_envs=1, seed=42)
    eval_env = VecFrameStack(eval_env, n_stack=4)

    # =================================================================
    # STEP 1: Evaluate trained agent
    # =================================================================
    print(f"\nEvaluating trained agent ({args.episodes} episodes)...")

    agent_rewards, agent_lengths = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=args.episodes,
        deterministic=args.deterministic,
        return_episode_rewards=True,
    )

    agent_mean = float(np.mean(agent_rewards))
    agent_std = float(np.std(agent_rewards))
    agent_min = float(np.min(agent_rewards))
    agent_max = float(np.max(agent_rewards))

    print(f"  Mean: {agent_mean:.1f} ± {agent_std:.1f}")
    print(f"  Min: {agent_min:.1f}, Max: {agent_max:.1f}")

    # =================================================================
    # STEP 2: Compute random baseline
    # =================================================================
    if args.skip_random:
        random_mean = BENCHMARKS["random_score"]
        random_std = 0.0
        random_rewards = [random_mean]
        print(f"\nUsing paper random baseline: {random_mean}")
    else:
        print(f"\nComputing random baseline ({args.episodes} episodes)...")

        # Create a dummy model that takes random actions
        random_env = make_atari_env("BreakoutNoFrameskip-v4", n_envs=1, seed=123)
        random_env = VecFrameStack(random_env, n_stack=4)

        random_rewards = []
        for ep in range(args.episodes):
            obs = random_env.reset()
            done = False
            total_reward = 0
            while not done:
                action = [random_env.action_space.sample()]
                obs, reward, done, info = random_env.step(action)
                total_reward += reward[0]
                if "episode" in info[0]:
                    total_reward = info[0]["episode"]["r"]
                    break
            random_rewards.append(total_reward)
            if (ep + 1) % 10 == 0:
                print(f"  Episode {ep + 1}/{args.episodes}")

        random_env.close()
        random_mean = float(np.mean(random_rewards))
        random_std = float(np.std(random_rewards))
        print(f"  Random baseline: {random_mean:.1f} ± {random_std:.1f}")

    # =================================================================
    # STEP 3: Calculate human-normalized score
    # =================================================================
    hns = compute_human_normalized_score(agent_mean, random_mean, BENCHMARKS["human_score"])

    # DeepMind's DQN HNS for comparison
    deepmind_hns = compute_human_normalized_score(
        BENCHMARKS["deepmind_dqn"], BENCHMARKS["random_score"], BENCHMARKS["human_score"]
    )

    # =================================================================
    # STEP 4: Load learning curve from training
    # =================================================================
    print("\nLoading learning curve data...")
    learning_curve = load_learning_curve("logs/evaluations.npz")
    print(f"  Found {len(learning_curve)} evaluation points")

    # =================================================================
    # STEP 5: Print summary
    # =================================================================
    print("\n" + "=" * 60)
    print("                 RESULTS SUMMARY")
    print("=" * 60)
    print(f"\n{'Metric':<30} {'Value':>12} {'Source':>15}")
    print("-" * 60)
    print(f"{'Our DQN Mean':<30} {agent_mean:>12.1f} {'Computed':>15}")
    print(f"{'Our DQN Std':<30} {agent_std:>12.1f} {'Computed':>15}")
    print(f"{'Our DQN Min':<30} {agent_min:>12.1f} {'Computed':>15}")
    print(f"{'Our DQN Max':<30} {agent_max:>12.1f} {'Computed':>15}")
    print("-" * 60)
    print(f"{'Random (Computed)':<30} {random_mean:>12.1f} {'Computed':>15}")
    print(f"{'Random (DeepMind)':<30} {BENCHMARKS['random_score']:>12.1f} {'Nature 2015':>15}")
    print(f"{'Human (DeepMind)':<30} {BENCHMARKS['human_score']:>12.1f} {'Nature 2015':>15}")
    print(f"{'DQN (DeepMind, 200M)':<30} {BENCHMARKS['deepmind_dqn']:>12.1f} {'Nature 2015':>15}")
    print("-" * 60)
    print(f"{'Human-Normalized Score':<30} {hns:>11.1f}% {'Computed':>15}")
    print(f"{'DeepMind DQN HNS':<30} {deepmind_hns:>11.1f}% {'Nature 2015':>15}")
    print("=" * 60)
    print(f"\nRef: {BENCHMARKS['citation']}")

    # =================================================================
    # STEP 6: Save JSON report
    # =================================================================
    report = {
        "timestamp": datetime.now().isoformat(),
        "model_path": args.model,
        "n_episodes": args.episodes,
        "deterministic": args.deterministic,
        "agent": {
            "mean": agent_mean,
            "std": agent_std,
            "min": agent_min,
            "max": agent_max,
            "all_rewards": [float(r) for r in agent_rewards],
            "all_lengths": [int(l) for l in agent_lengths],
        },
        "random_baseline": {
            "mean": random_mean,
            "std": random_std,
            "computed": not args.skip_random,
        },
        "benchmarks": BENCHMARKS,
        "human_normalized_score": hns,
        "learning_curve": learning_curve,
    }

    report_path = output_dir / "metrics_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, cls=NumpyEncoder)
    print(f"\nReport saved: {report_path}")

    # =================================================================
    # STEP 7: Generate plots
    # =================================================================
    setup_plots()

    # Plot 1: Learning Curve
    if learning_curve:
        fig, ax = plt.subplots()
        timesteps = [p["timestep"] / 1e6 for p in learning_curve]
        means = [p["mean"] for p in learning_curve]
        stds = [p["std"] for p in learning_curve]

        ax.plot(timesteps, means, 'b-', linewidth=2, label='Mean Score')
        ax.fill_between(timesteps,
                        np.array(means) - np.array(stds),
                        np.array(means) + np.array(stds),
                        alpha=0.2, label='±1 Std')
        ax.axhline(BENCHMARKS["human_score"], color='g', linestyle='--',
                   label=f'Human ({BENCHMARKS["human_score"]})')
        ax.axhline(BENCHMARKS["random_score"], color='r', linestyle=':',
                   label=f'Random ({BENCHMARKS["random_score"]})')

        ax.set_xlabel('Training Timesteps (Millions)')
        ax.set_ylabel('Average Score')
        ax.set_title('DQN Learning Curve on Atari Breakout')
        ax.legend(loc='upper left')
        plt.savefig(output_dir / "learning_curve.png")
        plt.close()
        print(f"Plot saved: {output_dir}/learning_curve.png")

    # Plot 2: Score Distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(agent_rewards, bins=min(20, len(agent_rewards)),
            edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(agent_mean, color='red', linewidth=2, label=f'Mean: {agent_mean:.1f}')
    ax.set_xlabel('Score')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Score Distribution ({args.episodes} episodes)')
    ax.legend()
    plt.savefig(output_dir / "score_distribution.png")
    plt.close()
    print(f"Plot saved: {output_dir}/score_distribution.png")

    # Plot 3: Comparison Bar Chart
    fig, ax = plt.subplots(figsize=(10, 6))
    categories = ['Random\n(Computed)', 'Random\n(DeepMind)', 'Human\n(DeepMind)',
                  'Our DQN', 'DQN\n(DeepMind)']
    values = [random_mean, BENCHMARKS["random_score"], BENCHMARKS["human_score"],
              agent_mean, BENCHMARKS["deepmind_dqn"]]
    colors = ['lightcoral', 'indianred', 'lightgreen', 'steelblue', 'navy']

    bars = ax.bar(categories, values, color=colors, edgecolor='black')
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{val:.1f}', ha='center', fontsize=10)

    ax.set_ylabel('Average Score')
    ax.set_title('Performance Comparison on Atari Breakout')
    plt.savefig(output_dir / "comparison_chart.png")
    plt.close()
    print(f"Plot saved: {output_dir}/comparison_chart.png")

    # Plot 4: Human-Normalized Score
    fig, ax = plt.subplots(figsize=(8, 5))
    categories = ['Random', 'Human', 'Our DQN', 'DQN (DeepMind)']
    values = [0, 100, hns, deepmind_hns]
    colors = ['lightcoral', 'lightgreen', 'steelblue', 'navy']

    bars = ax.bar(categories, values, color=colors, edgecolor='black')
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                f'{val:.1f}%', ha='center', fontsize=11, fontweight='bold')

    ax.axhline(100, color='green', linestyle='--', alpha=0.5)
    ax.set_ylabel('Human-Normalized Score (%)')
    ax.set_title('Human-Normalized Performance')
    plt.savefig(output_dir / "human_normalized.png")
    plt.close()
    print(f"Plot saved: {output_dir}/human_normalized.png")

    eval_env.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
