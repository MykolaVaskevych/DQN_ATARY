import json
from pathlib import Path
import matplotlib.pyplot as plt
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


def load_tb_json(filename):
    path = BASE_DIR / filename
    with open(path, "r") as f:
        data = json.load(f)

    # TensorBoard export format: [timestamp, step, value]
    steps = [entry[1] for entry in data]
    values = [entry[2] for entry in data]

    # Convert steps to millions for readability
    steps = [s / 1e6 for s in steps]

    return steps, values


def plot_training_curves():
    steps_rew, rew = load_tb_json("rollout_ep_rew_mean.json")
    steps_len, ep_len = load_tb_json("rollout_ep_len_mean.json")

    plt.figure(figsize=(10, 6))
    plt.plot(steps_rew, rew, label="Mean Episode Reward")
    plt.plot(steps_len, ep_len, label="Mean Episode Length")
    plt.xlabel("Training Timesteps (Millions)")
    plt.ylabel("Value")
    plt.title("Training Rollout Metrics")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(BASE_DIR / "training_curves.png", dpi=300)
    plt.close()


def plot_exploration_rate():
    steps_eps, eps = load_tb_json("rollout_exploration_rate.json")

    plt.figure(figsize=(10, 4))
    plt.plot(steps_eps, eps)
    plt.xlabel("Training Timesteps (Millions)")
    plt.ylabel("Epsilon")
    plt.title("Exploration Rate (Epsilon-Greedy)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(BASE_DIR / "exploration_rate.png", dpi=300)
    plt.close()


def plot_evaluation_curves():
    steps_eval_rew, eval_rew = load_tb_json("eval_mean_reward.json")
    steps_eval_len, eval_len = load_tb_json("eval_mean_ep_length.json")

    plt.figure(figsize=(10, 6))
    plt.plot(steps_eval_rew, eval_rew, marker="o", label="Eval Mean Reward")
    plt.xlabel("Training Timesteps (Millions)")
    plt.ylabel("Mean Reward")
    plt.title("Evaluation Performance During Training")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(BASE_DIR / "learning_curve.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(steps_eval_len, eval_len, marker="o", label="Eval Mean Episode Length")
    plt.xlabel("Training Timesteps (Millions)")
    plt.ylabel("Mean Episode Length")
    plt.title("Evaluation Episode Length During Training")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(BASE_DIR / "eval_episode_length.png", dpi=300)
    plt.close()


def main():
    plot_training_curves()
    plot_exploration_rate()
    plot_evaluation_curves()
    print("Plots saved to:", BASE_DIR)


if __name__ == "__main__":
    main()
