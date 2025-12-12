"""
DQN Training Script for Atari Breakout

This script trains a Deep Q-Network (DQN) agent on the Atari Breakout game
using Stable Baselines 3 and Gymnasium.
"""

# Import ALE (Arcade Learning Environment) - registers Atari ROMs with Gymnasium
import ale_py

# Import DQN algorithm - Deep Q-Network that learns Q-values using neural networks
from stable_baselines3 import DQN

# Import helper to create Atari envs with standard preprocessing (grayscale, resize, frame skip)
from stable_baselines3.common.env_util import make_atari_env

# Import wrapper to stack consecutive frames - lets network see motion/velocity
from stable_baselines3.common.vec_env import VecFrameStack

# Import callbacks - functions called during training for checkpointing and evaluation
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback


def main():
    # ==================== ENVIRONMENT SETUP ====================

    # Create vectorized Atari environment with preprocessing applied
    vec_env = make_atari_env(
        # Use NoFrameskip version - make_atari_env adds its own frame skipping
        "BreakoutNoFrameskip-v4",
        # Run 8 environments in parallel - RTX 4090 handles this easily
        # More envs = faster experience collection (~2x speedup vs 4 envs)
        n_envs=8,
        # Set random seed for reproducibility
        seed=42,
    )

    # Stack 4 consecutive frames together (84x84x1 -> 84x84x4)
    # This lets the CNN perceive motion - single frame has no velocity info
    vec_env = VecFrameStack(vec_env, n_stack=4)

    # Create separate environment for periodic evaluation during training
    # Uses n_envs=1 because evaluation doesn't need parallelism
    eval_env = make_atari_env("BreakoutNoFrameskip-v4", n_envs=1, seed=123)
    # Must apply same frame stacking as training env
    eval_env = VecFrameStack(eval_env, n_stack=4)

    # ==================== CALLBACKS SETUP ====================

    # Callback to save model checkpoints periodically (insurance against crashes)
    checkpoint_callback = CheckpointCallback(
        # Save every 500k timesteps (20 checkpoints for 10M steps)
        save_freq=500_000,
        # Directory to save checkpoints
        save_path="./checkpoints/",
        # Filename prefix for saved models
        name_prefix="dqn_breakout",
        # Also save replay buffer - needed to truly resume training
        save_replay_buffer=True,
        # Save normalization stats if environment uses them
        save_vecnormalize=True,
    )

    # Callback to evaluate model and save the best performing one
    eval_callback = EvalCallback(
        # Environment to evaluate on
        eval_env,
        # Directory to save best model
        best_model_save_path="./best_model/",
        # Directory to save evaluation logs
        log_path="./logs/",
        # Evaluate every 100k timesteps (100 evaluations for 10M steps)
        eval_freq=100_000,
        # Use greedy actions (no exploration) during evaluation
        deterministic=True,
        # Don't render during evaluation - would slow it down
        render=False,
    )

    # ==================== DQN MODEL SETUP ====================

    # Initialize DQN model with Atari-optimized hyperparameters
    # Tuned for RTX 4090 (24GB VRAM) + 32GB RAM at 80% safe utilization
    # Based on original DQN paper (Mnih et al., 2015) and SB3 recommendations
    model = DQN(
        # Use CNN policy - required for image observations (84x84x4 frames)
        "CnnPolicy",
        # Pass the training environment
        vec_env,
        # Learning rate for Adam optimizer - standard for Atari DQN
        learning_rate=1e-4,
        # Replay buffer size - 400k uses ~22GB RAM (80% of 27.5GB available)
        # SB3 reports 56GB/1M, so 400k ≈ 22GB, safely under 80% threshold
        buffer_size=400_000,
        # Start learning after 50k steps - need diverse experiences first
        # This is the Atari default from original DQN paper
        learning_starts=50_000,
        # Batch size 64 - RTX 4090 easily handles this (uses ~500MB VRAM)
        # Larger batches = more stable gradients = faster convergence
        batch_size=64,
        # Target network update coefficient (1.0 = hard copy)
        tau=1.0,
        # Discount factor - how much to value future vs immediate rewards
        gamma=0.99,
        # Perform gradient update every 4 environment steps
        train_freq=4,
        # Number of gradient steps per update
        gradient_steps=1,
        # Copy online network to target network every 10k steps
        # This stabilizes training by keeping target Q-values fixed longer
        target_update_interval=10_000,
        # Decay epsilon over 10% of training (for 1M steps = 100k steps of decay)
        # Original paper decayed over 1M frames
        exploration_fraction=0.1,
        # Start with 100% random actions (full exploration)
        exploration_initial_eps=1.0,
        # End with 1% random actions (mostly exploitation)
        exploration_final_eps=0.01,
        # Print training progress to console
        verbose=1,
        # Directory for TensorBoard logs (visualize training curves)
        tensorboard_log="./tensorboard_logs/",
        # Use CUDA for GPU acceleration (auto-detected, but explicit is clearer)
        device="cuda",
    )

    # ==================== TRAINING ====================

    # Print training info
    print("Starting DQN training on Atari Breakout...")
    print(f"Environment: BreakoutNoFrameskip-v4")
    # Show observation shape - should be (84, 84, 4) after preprocessing
    print(f"Observation space: {vec_env.observation_space}")
    # Show action space - Breakout has 4 actions: NOOP, FIRE, LEFT, RIGHT
    print(f"Action space: {vec_env.action_space}")

    # Total environment steps to train for
    # 10M steps matches original DQN paper - takes ~2 hours on RTX 4090
    # For longer overnight training (8-10 hours), use 40-50M steps
    # EvalCallback saves best model, so no risk of losing peak performance
    total_timesteps = 10_000_000

    # Start the training loop
    model.learn(
        # Number of environment steps to train for
        total_timesteps=total_timesteps,
        # List of callbacks to run during training
        callback=[checkpoint_callback, eval_callback],
        # Show progress bar with ETA
        progress_bar=True,
    )

    # ==================== SAVE AND CLEANUP ====================

    # Save the final trained model (creates dqn_breakout_final.zip)
    model.save("dqn_breakout_final")
    # Confirm training completed
    print("Training complete! Model saved as 'dqn_breakout_final.zip'")

    # Close environments to free resources (memory, window handles)
    vec_env.close()
    eval_env.close()


# Standard Python entry point - only run main() when script is executed directly
if __name__ == "__main__":
    main()
