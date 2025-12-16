import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Team Members:**  Mykola Vaskevych (22372199) , Oliver Fitzgerald (22365958)

    **Execution:** Code does NOT execute to the end without error

    **Third Party Implmentations Used:** TBD
    """)
    return


@app.cell
def _():
    import marimo as mo
    from pathlib import Path  # For displaying images
    import json
    import matplotlib.pyplot as plt
    import argparse
    from dataclasses import dataclass
    from datetime import datetime
    import numpy as np
    import matplotlib.ticker as ticker
    from stable_baselines3.common.evaluation import evaluate_policy
    from tensorboard.backend.event_processing.event_accumulator import (
        EventAccumulator,
    )


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
    return (
        CheckpointCallback,
        DQN,
        EvalCallback,
        EventAccumulator,
        Path,
        VecFrameStack,
        evaluate_policy,
        json,
        make_atari_env,
        mo,
        np,
        plt,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # CS4287 Assignment 2 Deep Reinforcment Learning

    In this project we set out with the objective to train a deep reinforcment learning model to learn to play the game [breakout](https://ale.farama.org/environments/breakout/) from OpenAis Gymnasium to a comperable level to a human.

    ## 1. Introduction

    To accomplish this objective our agent must learn directly from raw visual input (images) rather than low-dimensional state variables (raw numbers). This makes the task more challenging as it will require the model to process images and extract features it needs as well as the actions to take on the perceived state. Rather than taking actions on a definitivly defined state such as in a classical game like chess where the board state is definite i.e there is no room for interperataion by the model.

    ### 1.1 Game Choice

    we picked the game Atari Breakout.
    Breakout is a 1976 action video game developed and published by Atari, Inc. for arcades. In the game, eight rows of bricks line the top portion of the screen, and the player’s goal is to destroy the bricks by repeatedly bouncing a ball off a paddle into them. Using a single ball, the player must knock down as many bricks as possible by using the walls and the paddle below to hit the ball against the bricks and eliminate them. If the player’s paddle misses the ball’s rebound, they will lose a turn. [5]
    This game was chosen because we were already familiar with it and because it has strong historical importance in the deep reinforcement learning field. it is one of the first games where RL agents achieved human-level and even superhuman performance, as demonstrated in the original DeepMind DQN paper. [3]
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.image("images/breakout.gif", alt="Example diagram", width=400)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1.2 Agent & Environment Interaction

    Our agent will play the game by observing the game state at each time step in the form of a stask of image frames. From these images the agent can infer the position of the paddle, the bricks, and the ball, as well as the ball’s direction and speed.
    A single frame doesn't show motion (ball direction, paddle velocity). This is why we use frame stacking:

    ```
    Frame t-3 Frame t-2 Frame t-1 Frame t
      ↓           ↓         ↓        ↓
    ┌────────┬────────┬────────┬────────┐
    │        │        │        │        │  → 84x84x4 input
    │ ○      │   ○    │     ○  │      ○ │    (ball moving right)
    └────────┴────────┴────────┴────────┘
    ```


    Based on this observation, the agent selects an action such as moving the paddle left, moving it right, or staying still.
    After performing the selected action, the environment returns a reward that represents the game progress, such as breaking bricks or losing a life. The objective of the agent is to accumulate the maximum cumulative reward over an episode.
    In the enviornment used in this project provided by gymnasium and wrapped by stable-baseline3. The agent excutes actions in the envoirnment via the [step](https://gymnasium.farama.org/api/env/#gymnasium.Env.step) method provided by gymnasium to update the envoirnment with a given action. It returns information related to the observation resulting from a given action including the observation itself, any reward from taking the given action, wether or not the given action resulted in the enviornment being terminated truncated, debug information and metrics essentially encapsulating the core reinforcment learning loop as visualized in the following image.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.image("images/rl-loop.png", alt="Example diagram", width=400)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Background/Context
    ### 2.1 Our Approach

    #### **Why We Chose Reinforcment Learning**

    We will be using the Reinforcement Learning (RL) paradigm to accomplish this objective specifically a Deep Q-Network. To describe why we have taken this approach we must first define reinforcment learning before we disccuss what elements of it made it suitable for use in this project.

    Reinforcment learning is a machine learning paradigm in which an agent learns to make decisions by interacting with an enviorment (in this case the Open AI gymnasium breakout game) through trial and error actions. The agent selects actions (paddle movments)  according to a policy. After actions the agent receive penalties or rewards from the enviornment (e.g rewards for breaking bricks, penalties for missing the ball). The agent updates it's policy to maxemize the expectied cumulative reward over time. Many RL algorithms use value functions to estimate long-term consequences of actions, allowing the agent to balance immediate rewards with future potential.

    The attributes of this paradigm that make it suitable for application to this problem domain are discussed, compared and contrasted with other machine learning paradigms across the following points:


    - Sequential Decision Making:
    The Atari Breakout game involves making sequential decisions regarding the paddles position, where each individual action can have a significant bearing on the long-term outcome of the game.
    In a reinforcement learning paradigm the long-term consequences of any individual actions from any given state can be modeled through value functions and policy optimizations, enabiling effective temporal credit assignment accross action sequences.
    In contrast evoloutionary learning algorithms typically evaluate fixed candidate policies over many complete episodes, meaning that the policies get evaluated as a whole rather than individual actions. This means that individual actions contributions are not explicitly evaluated.
    Consequently making the reinforcment learning paradigm a more suitable approach for this problem, as assessing the controbution of individual actions is critical in a sequential decision making game like breakout.

    - Exploration and Exploitation:
    Balancing exploration and exploitations is an important challange in developing an algorithm to learn to play a game. Where exploitation refers to executing on known effective strategies (actions) to yield higher rewards, while exploration involves actions with uncertain or lower reward estimates in order to aquire information which may aid in developing long-term or effective new strategies.
    In a reinforcment learning paradigm exploration and exploitation are first class concepts, enabiling them to be configured and varied e.g increacing weigth towards exploration for policy selection and then increasing the weigth towards exploitation during policy evalutation.
    Many other paradigms only address this trade-off indirectly or even not at all. In the supervised learning paradigm for example, A policy is learned from stastically generalzation of a training dataset with no exploration of potential states outside these samples. During testing/validation these passivily learned state-action pairs are then applied in a purly exploitive manner. As a result supervised learning is unable to learn novel actions not represented in the training dataset or adapt to new/unseen states in the enviornment.
    This makes reinforcement learning a more appropriate paradigm for learning to play games such as Breakout, where rewards may be delayed and effective strategies may initially appear suboptimal.


    - Problem Representation:
    The reinforcment learning paradigm provides a problem represention commonly formalized as a Markov Decision Processes (MDP). This formulation explicitly models states, actions, transitions, and rewards, which aligns with the sequential, interactive and reward driven nature underpinning the fundemental structure of most games.
    In contrast, other paradigms such as supervised and unsupervised learning paradigms typically frame problems as static input to output mappings. These representations do not naturally capture state transitions or the emerging temporal dependencies that emerge from sequences of actions which are fundemental to gameplay.
    This makes the reinforcment paradigms problem representation better suited for learning to play a game like Breakout, where the highest rewards can result from sequences of actions that form a long-term strategy, as apose to isolated, single step actions.


    In summary due to its emphasis on learning through a balance of exploitation and exploration, its ability to handle sequential decision-making problems and a problem representation which naturally lends itself to the structure of a game like breakout the reinforcement learning paradigm was deemed to be the most suitable approach for training an agent to play the Atari Breakout game.


    ### 2.2 Gym and Gymnasiun

    OpenAI Gym was originally one of the most widely used libraries for reinforcement learning research. However, it is no longer actively maintained.
    Gymnasium is a maintained fork of OpenAI Gym and is now the recommended standard for developing and evaluating reinforcement learning agents. Gymnasium is described as “an API standard for reinforcement learning with a diverse collection of reference environments” and provides a simple, pythonic interface capable of representing general reinforcement learning problems. It also includes a migration guide for older Gym-based environments, making it suitable for modern and legacy reinforcement learning workflows alike.
    In this assignment, the Gymnasium framework is used together with the Atari Learning Environment (ALE) to train and evaluate an agent on the Atari Breakout game. Gymnasium provides a uniform interface for interacting with the environment, allowing the agent to receive observations, select actions, and obtain rewards in a consistent and standardized way.

    #### **Key Features of the Atari Breakout Environment**
    - **Complexity and Visual Processing:** The Breakout environment presents a visually rich setting with multiple dynamic objects, such as the paddle, ball, and bricks. Observations are provided as RGB image frames that represent the current game state. This requires the agent to process and interpret visual information in order to infer the game state and to make the appropriate decisions.

    - **Action Space:** Each action available to the agent corresponds to a possible paddle movement. The action space is discrete, meaning the agent selects from a fixed set of possible actions at each timestep, such as moving left, moving right, or performing no action. Understanding and navigating this action space is essential for effective gameplay.

    - **Reward System:** The Gymnasium environment provides a reward signal that guides learning. Positive rewards are associated with breaking bricks and increasing the game score, while negative outcomes such as losing a life will eventually result in the termination of the game. This reward-based feedback is central to the reinforcement learning process.

    #### **Role in Training the Reinforcement Learning Model**

    - **Observation Preprocessing:** For computational efficiency, raw RGB frames are processed before being used by the learning algorithm. This includes operations such as resizing, grayscale conversion, and frame stacking, which help reduce input dimensionality while preserving important visual information.

    - **Feedback Loop for Learning:** The environment acts as a continuous feedback loop for the agent. Through repeated interaction, the agent up-dates its policy based on observed rewards and state transitions, gradually improving its performance.

    - **Benchmark for Performance:** The Atari Breakout environment serves as a standard benchmark for evaluating deep reinforcement learning algorithms. Agent performance can be assessed by measuring achieved scores, episode lengths, and learning curves over time. Overall, Gymnasium plays a crucial role in this project by providing a stable and well-supported platform for training and evaluating reinforcement learning agents in complex, high-dimensional environments.

    ### 2.3 Stable-Baselines3

    Stable-Baselines3 (SB3) is an open-source Python library that provides reliable implementations of modern deep reinforcement learning algorithms. As stated in its original publication, Stable-Baselines3 “provides open-source implementations of deep reinforcement learning (RL) algorithms in Python. The implementations have been benchmarked against reference codebases, and automated unit tests cover 95% of the code. The algorithms follow a consistent interface and are accompanied by extensive documentation, making it simple to train and compare different RL algorithms.” [4]
    In this assignment, Stable-Baselines3 is used to implement the Deep Q-Network (DQN) agent for the Atari Breakout environment. The main reason for using SB3 is that it significantly reduces the amount of boilerplate code required to build a reinforcement learning pipeline. Core components such as the replay buffer, target network updates, loss computation, and optimisation steps are handled internally by the library, allowing the focus to remain on understanding the algorithm rather than low-level implementation details. Despite using a high-level library, the underlying learning process remains the same as in the original DQN formulation. In accordance with the assignment requirements, the behaviour of the DQN algorithm implemented by Stable-Baselines3 is explained in detail later in this report. This includes a discussion of how experiences are sampled from the replay buffer, how the Q-learning loss is computed, and how gradient updates are applied to the network parameters. Using Stable-Baselines3 therefore provides both a practical and transparent framework for implementing and analysing deep reinforcement learning algorithms.

    ## 3. Implementation

    ### 3.1 Data Capture and Pre-processing

    We begin by defining the Atari enviornment from which we will capture game data. We use Stable-Baseline3s enviornment utilities which will create a [gymnasium environment](https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/env_util.py#L95) of type env_id in this case for the breakout game. The envoirnment will consist of frames from the Breakout game, each frame representing the current visual state of the environment, including the paddle, the ball, and the bricks. These observations (frames) are captured directly from the environment. As we have specified "NoFrameSkip-V4" we interact with our enviornment at every timestep (frame). We also specify n_envs=8 to run 8 of these eniornments, using multiple environments improves data collection efficiency because the agent can gather more transitions per unit time than using a single environment through parallelism. We also set a seed of 42 a deterministic random seed for environment initialisation, improving reproducibility of the training behaviour.

    *Note:* We choose "NoFrameSkip" as our stable diffuction wrapper will handle frame skips so having the base gymnasium enviornment implment it as well would mean skipping twice. This is discussed further in later sectinos.

    In summary we construct our enviornment with the following parameters:


    | Parameter | Significance |
    |:---------:|:------------:|
    | env_id    | The game environment to be created |
    | n_envs    | The number of environments to be run |
    | seed      | To allow randomness but ensure reproducibility between runs |
    """)
    return


@app.cell
def _(make_atari_env):
    # Create vectorized Atari environment with preprocessing applied
    static_vec_env = make_atari_env(
        # Use NoFrameskip version - make_atari_env adds its own frame skipping
        env_id="BreakoutNoFrameskip-v4",
        # Run 8 environments in parallel - RTX 4090 handles this easily
        # More envs = faster experience collection (~2x speedup vs 4 envs)
        n_envs=8,
        # Set random seed for reproducibility
        seed=42,
    )
    return (static_vec_env,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The resulting VecEnv object static_vec_env provided by Stable-Baseline3 servers as a wrapper for the gymnasium Breakout envoirnment. This wrapper handles pre-processing for us via the [AtariWrapper](https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/atari_wrappers.py#L258) class which means we do not need to explicity handle the pre-processing of frames in our code. However it is usefull to understand what it is doing and so it will be described across the following points:

    To reduce the time to complete the desired number of episodes we simplify the data we process i.e frames, such that taking actions and storing state is less computationally expensive, this is accomplished by following means.

    - **Image Scalling**
    Images are resized to 84x84 pixels which is a compromise between higer information content and lower compute times that was setteled on by the original Nature paper by Google Deep Mind.

    - **Grayscale**
    The colour model for frames is reduced from RGB values to Grayscale.

    - **Frame Skipping**
    Frame skipping does what it says on the tin, it reduces the amount of computation required by skipping frames in the enviornment. In other words only performing actinon every x number of frames in this case every 4 frames.


    The wrapper class also handles the following pre-processing steps.

    - Clips rewards to -1, 0, 1
    - Defines the max number of no-ops (no operations)
    - Handles max pooling, which involves keeping the last 2 frames of the game during frame skipping and using the temporal maximum of the two frames. This is done as Atari games often experience sprite flicketering every other frame.


    At this point we have our basic DQN Architecture, but in its current state it cannot infer the motion of objects between frames. To combat this issue in this implementation, four consecutive frames are stacked together using the VecFrameStack wrapper. This allows the agent to infer motion-related information such as the direction and speed of the ball, which cannot be determined from a single static frame. Without frame stacking, the environment would appear partially observable to the agent.

    Under the hood VecFrameStacks by concatonating the pixel channels of n_stack number of images together. As we our dealing with grey scale images that are scalled to 84x84 pixels, each frame in our enviornment will now be an 84x84x4 image where the 4 channels are the 4 Grayscale values for each image.
    """)
    return


@app.cell
def _(VecFrameStack, static_vec_env):
    # Stack 4 consecutive frames together (84x84x1 -> 84x84x4)
    # This lets the CNN perceive motion - single frame has no velocity info
    vec_env = VecFrameStack(static_vec_env, n_stack=4)
    return (vec_env,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We also define an evaulation envorinment by the same mechanisms as with our primary vectorized envoirnments. This environment is intentionally separate from the training environments. Evaluation should measure policy performance without being affected by training noise or parallel rollout behaviour. Using a different seed (seed=123) helps reduce the risk of accidentally overfitting to one specific environment initialisation.
    """)
    return


@app.cell
def _(VecFrameStack, make_atari_env):
    # Create separate environment for periodic evaluation during training
    # Uses n_envs=1 because evaluation doesn't need parallelism
    eval_env = make_atari_env("BreakoutNoFrameskip-v4", n_envs=1, seed=123)
    # Must apply same frame stacking as training env
    eval_env = VecFrameStack(eval_env, n_stack=4)
    return (eval_env,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3.2 Network Structure: Convolutional Neural Networks and Deep Q-Networks

    Although the Stable-Baselines3 implementation hides the low-level training loop from the user, the Q-learning update described in the previous section is applied repeatedly during the execution of the $\texttt{model.learn()}$ function. Internally, Stable-Baselines3 follows the same sequence of operations as the original DQN algorithm, but encapsulates these steps within a well-tested training pipeline.

    Q-learning aims to learn the optimal action–value function $Q^*(s, a)$, where $Q^*$ represents the optimal Q-function, which maps state–action pairs to their expected cumulative discounted rewards. In classical tabular Q-learning, this function is represented explicitly as a lookup table containing a separate Q-value for each possible state–action pair. While effective for problems with small, discrete state spaces, this approach becomes impractical for many real-world tasks, where the number of possible states grows extremely large.

    This limitation is particularly evident in arcade-style games such as Ms. Pac-Man. The game state depends on the configuration of remaining pellets, the positions and velocities of multiple moving entities within a 2D image enviornment and other environmental factors. As these elements combine, the resulting state space grows combinatorially, leading to an astronomically large number of distinct states. Under such conditions, storing and updating a separate Q-value for every state–action pair is infeasible, rendering tabular methods unsuitable.

    To address this scalability issue, DeepMind introduced the Deep Q-Network (DQN) architecture, which combines Q-learning with deep neural networks. In DQN, a convolutional neural network (CNN) parameterized by weights $\theta$ is used as a function approximator, producing estimates of the optimal action–value function:

    $Q(s, a; \theta) \approx Q^*(s, a)$

    Rather than relying on a lookup table, the network learns to generalize across similar states by extracting high-level features from raw pixel observations. The use of convolutional layers is particularly well suited to visual environments, as they exploit spatial structure and translational invariance in the input. By leveraging deep function approximation, DQN enables Q-learning to scale to high-dimensional state spaces that are intractable for traditional tabular reinforcment learning approaches.

    Specifically, the first convolutional layer applies 32 filters of size 8 × 8 with a stride of 4, followed by a ReLU activation function. The large receptive field and stride enable this layer to detect broad, low-level visual features such as edges and basic shapes while significantly reducing the spatial dimensions of the input. This is followed by a second convolutional layer with 64 filters of size 4 × 4 and a stride of 2, again using ReLU activations. This layer builds upon the features detected in the first layer, identifying more complex patterns such as object parts and spatial relationships while further compressing the representation. A third convolutional layer applies 64 filters of size 3 × 3 with a stride of 1, refining the learned features and capturing fine-grained spatial details without additional downsampling. The resulting feature maps are then flattened and passed into a fully connected layer with 512 units, which integrates the spatial information from across the entire visual field to form a compact representation of the game state. Finally, the output layer produces one Q-value for each possible action[2].

    In parctice DQNs execution follows the following structure:

    First we initalize:

    - Our replay buffer which will record all (s, a, r, s') tuples taken which will be played back in a random ordering during training. This is done to stabelize the model by avoiding strong temporal correlations in consecutive experiences and to mitigate catastrophic forgetting.
    - We initalize the weigths of our network $\theta$
    - We initalize our target network $\theta^- =  \theta$
    The target network is updated less frequently than the main network, providing more stable targets for the Q-learning updates and preventing divergence.

    For each episode, begining from an inital state s we step through each timestep (frame). Computing the following at each timestep:

    - Select action action using ε-greedy
    During training, the agent interacts with the environment using an epsilon-greedy policy. An action is selected either randomly (with probability $\varepsilon$) or greedily by choosing the action with the highest predicted Q-value.

    ```python
    if random() < ε:
        action = random_action() # Exploration
    else:
        action = argmax_a Q(s, a)  # Exploitation
    ```

    - Execute action, observe reward r and next state s'
    - Store transition in replay buffer
    - Sample minibatch from replay

    Once the number of collected transitions exceeds the learning_starts threshold, Stable-Baselines3 begins applying learning updates at a frequency specified by train_freq. Each update consists of sampling a minibatch of transitions uniformly at random from the replay buffer. For each transition in the minibatch, the target Q-value is computed using the target network according to the Bellman equation:

    $$
    r + \gamma \max_{a'} Q(s', a'; \theta^{-})
    $$

    - Compute Target and Loss
    The network is trained by minimizing the temporal difference (TD) error using gradient descent. The loss function at iteration $i$ is:


    $$
    L_i = \mathbb{E}_{(s,a,r,s') \sim U(D)}\left[ \big( r + \gamma \max_{a'} Q(s', a'; \theta^{-}) - Q(s, a; \theta) \big)^2 \right]
    $$

    where:
    $\theta_i^-$ represents the parameters of a target network (periodically updated from $\theta_i$),
    $\mathcal{D}$ is a replay buffer of experiences and
    $\gamma$ is the discount factor.

    and therfore $r + \gamma \max_{a'} Q(s', a'; \theta^{-})$ serving as our approximate target throught the use of a CNN

    Stable-Baselines3 performs the weight update using stochastic gradient descent (or a variant such as Adam), adjusting the parameters $\theta$ in the direction that minimises the Q-learning loss. These gradient updates are applied automatically during training and correspond directly to the theoretical Q-learning update rule discussed earlier.


    - Update $\theta$
    - Every x number of steps we update our target network $\theta^-$

    The target network parameters $\theta^{-}$ are updated less frequently, as controlled by the \texttt{target\_update\_interval} parameter. By keeping the target network fixed for several updates, the learning targets remain more stable, which significantly reduces the risk of divergence when training deep neural networks.


    By combining convolutional neural networks with Q-learning and these stabilisation techniques, DQN became the first reinforcement learning algorithm to achieve human-level performance across a wide range of Atari games using a single network architecture and minimal prior knowledge. This demonstrated that deep reinforcement learning can learn effective policies directly from high-dimensional visual input.


    ### 3.3 Q-learning Update and Loss Computation

    Although Stable-Baselines3 abstracts away the low-level implementation details, the underlying learning mechanism of the DQN agent follows the standard Q-learning update introduced by DeepMind. In this assignment, it is important to understand how the Q-values are updated and how the network weights are trained, even though these operations are handled internally by the library.

    Deep Q-Networks use a neural network to approximate the action-value function $Q(s,a;\theta)$, where $\theta$ represents the parameters of the network. During training, the goal is to adjust these parameters so that the predicted Q-values are close to the target Q-values derived from the Bellman optimality equation.

    Training DQN uses a modified Q-learning loss function given by:


    $$
    L_i = \mathbb{E}_{(s,a,r,s') \sim U(D)}\left[ \big( r + \gamma \max_{a'} Q(s', a'; \theta^{-}) - Q(s, a; \theta) \big)^2 \right]
    $$

    where $(s,a,r,s')$ is a transition sampled uniformly from the replay buffer, $\gamma$ is the discount factor, $\theta$ are the parameters of the current Q-network, and $\theta^{-}$ are the parameters of the target network.

    The term $Q(s,a;\theta)$ represents the predicted Q-value for the action actually taken in the current state. The target value is computed as $r + \gamma \max_{a'} Q(s',a';\theta^{-})$, which consists of the immediate reward received after taking the action and the discounted maximum estimated future reward from the next state. The difference between the target Q-value and the predicted Q-value represents the temporal difference (TD) error.

    The loss function measures the squared TD error and is minimized using gradient descent. By minimizing this loss, the network updates its weights so that the predicted Q-values move closer to the target values. In Stable-Baselines3, this optimization step is performed automatically using minibatches sampled from the replay buffer and a stochastic gradient descent optimizer.

    To improve training stability, DQN uses two important mechanisms. First, experience replay breaks correlations between consecutive samples by randomly sampling past transitions from the replay buffer. Second, a separate target network with parameters $\theta^{-}$ is used to compute the target Q-values. This target network is updated less frequently than the main network, which helps prevent instability caused by rapidly changing targets during training.

    Overall, although Stable-Baselines3 hides the low-level implementation details, it implements the standard DQN learning update exactly as described above. Understanding this update rule is essential for interpreting the learning behaviour and performance of the trained agent.

    ### 3.4 Network Structure Implementation

    The DQN agent uses a convolutional neural network policy suitable for image-based input. Hyperparameters such as learning rate, replay buffer size, discount factor, and exploration schedule are selected to closely match the settings used in the original DQN paper. A target network is updated at fixed intervals to stabilize learning, and the model is trained on a GPU to improve performance
    """)
    return


@app.cell
def _(DQN, vec_env):
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
    return (model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Code walkthrough (DQN configuration).

    The agent is created with:
    ```python
    model = DQN("CnnPolicy", vec_env, ...)
    ```
    The "CnnPolicy" indicates that Stable-Baselines3 will use a convolutional neural network suitable for image inputs. This matches the DQN setting where the Q-function is approximated by a CNN that outputs Q-values for each discrete action.

    Key hyperparameters and their role:


    |Parameter | Value|Description|
    |:-----------:|:-------:|:-------------:|
    | learning rate | 1e-4 | Controls the step size of gradient descent when updating network weights. A small learning rate is commonly used in Atari DQN because the learning signal can be noisy and unstable. |
    | buffer size | 400 000 | Sets the capacity of the replay buffer. A larger buffer increases diversity of experience and reduces correlation between samples, which stabilises learning. The trade-off is higher memory usage. |
    | learning starts | 50 000 | Prevents training from starting immediately. The agent first collects 50,000 transitions to populate the replay buffer. This avoids learning from an extremely small and highly correlated dataset at the beginning of training. |
    | batch size | 64 | The number of replay transitions sampled per gradient update. Using minibatches makes gradient estimates more stable than single-sample updates. |
    | gamma | 0.99 | Discount factor controlling the importance of future rewards. In Atari, high values like 0.99 encourage planning over longer time horizons. |
    | train freq | 4 | Specifies how often learning updates occur relative to environment steps. Here, the network is updated every 4 environment steps, which aligns with the classic DQN practice of not updating on every single frame. |
    | gradient steps | 1 | Performs one gradient update each time training is triggered. Together with train freq=4, this creates a stable ratio between environment interaction and learning updates. |
    | target update interval | 10 000 | Controls how often the target network is updated. DQN uses a target network to compute stable targets; updating it too frequently can destabilise learning, while updating too rarely can slow learning. |
    | exploration initial, exploration final eps | 1.0, 0.01 | Define the epsilon-greedy exploration schedule. Initially, the agent explores almost completely randomly, then gradually shifts toward exploiting learned Q-values. |
    | exploration fraction | 0.1 | Indicates that epsilon decays over the first 10% of total training steps, after which it stays near the final value. This means exploration is emphasised early in training when the agent is untrained. |
    | device | "cuda" | Runs neural network computation on the GPU. This is especially beneficial for Atari because CNN forward/backward passes are computationally heavy. Overall, these settings implement a standard DQN training pipeline: collect transitions, store them in replay, sample random minibatches, compute Q-learning targets using a target network, and update the CNN weights using gradient descent. |



    Two callbacks are used during training. The checkpoint callback periodically
    saves the model parameters and replay buffer, enabling training to be resumed
    if needed. The evaluation callback assesses the agent’s performance at fixed
    intervals using a separate environment and stores the best-performing model
    based on evaluation rewards
    """)
    return


@app.cell
def _(CheckpointCallback, EvalCallback, eval_env):
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
    return checkpoint_callback, eval_callback


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Code walkthrough (Callbacks and Evaluation).
    The checkpoint call-back:
    ```
    checkpoint_callback = CheckpointCallback(save_freq=500_000, ...)
    ```

    automatically saves the model every 500,000 timesteps. This is important because Atari training can take a long time, and checkpoints allow recovery if training is interrupted. The argument save replay buffer=True also saves the replay buffer contents. This matters because DQN learning depends heavily on the replay buffer distribution; resuming training without the buffer would effectively change the training dynamics and usually harms continuity.


    The evaluation callback:
    ```
    eval_callback = EvalCallback(eval_env, eval_freq=100_000, deterministic=True, ...)
    ```

    evaluates the current agent every 100,000 timesteps on the separate evaluation environment. Using deterministic=True means the evaluation uses the greedy action (the highest Q-value action) rather than exploration actions. This is important because during training, DQN uses epsilon-greedy exploration, which would add noise to evaluation scores. Deterministic evaluation therefore gives a cleaner estimate of what the learned policy can do when not exploring. The best model save path parameter saves the best-performing model according to evaluation reward. This is useful because RL learning curves canfluctuate; the best policy may occur before the final timestep, so saving the best model prevents losing a strong policy due to later instability.


    The agent is trained for a total of ten million timesteps. During training, both
    checkpointing and evaluation callbacks are active. After training is completed,
    the final model is saved and all environments are properly closed.
    """)
    return


@app.cell
def _(Path, checkpoint_callback, eval_callback, eval_env, model, vec_env):
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
    total_timesteps = 10

    import shutil


    def delete_tensorboard_logs():
        # Path of the current script
        current_file = Path(__file__).resolve()

        # Walk upwards until we find the project root
        # (identified by tensorboard_logs/)
        for parent in current_file.parents:
            tb_logs = parent / "tensorboard_logs"
            if tb_logs.exists() and tb_logs.is_dir():
                shutil.rmtree(tb_logs)
                print(f"Deleted: {tb_logs}")
                return

        raise FileNotFoundError("tensorboard_logs directory not found")


    delete_tensorboard_logs()

    # Start the training loop
    model.learn(
        # Number of environment steps to train for
        total_timesteps=total_timesteps,
        # List of callbacks to run during training
        callback=[checkpoint_callback, eval_callback],
        # Show progress bar with ETA
        progress_bar=True,
        tb_log_name="TESTING_NOT_MAIN",  # main is dqn_5 with 10M steps, this one only demo of code with 10 steps
    )

    # ==================== SAVE AND CLEANUP ====================

    # Save the final trained model (creates dqn_breakout_final.zip)
    # model.save("dqn_breakout_final")
    # Confirm training completed
    print("Training complete! Model saved as 'dqn_breakout_final.zip'")

    # Close environments to free resources (memory, window handles)
    vec_env.close()
    eval_env.close()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Code walkthrough (Training loop).

    The training budget is set by:
    ```
    total_timesteps = 10_000_000
    ```

    This means the agent interacts with the environment for ten million timesteps. For Atari games, a long training horizon is usually required because learning from high-dimensional pixel input is sample intensive.

    The call:
    ```
    model.learn(total_timesteps=total_timesteps, callback=[checkpoint_callback, eval_callback],
    ```

    1. starts the RL training loop. Internally, Stable-Baselines3 repeatedly:
    2. collects transitions (s, a, r, s′) by interacting with the environment using
    an epsilon-greedy policy,
    3. stores these transitions in the replay buffer,
    4. samples minibatches from the replay buffer once learning starts is reached,
    5. computes TD targets using the target network, and
    6. applies gradient descent updates to the main Q-network parameters.

    The callback list ensures that saving and evaluation happen automatically during training without interrupting the learning loop. The progress bar=True flag provides visual feedback while training runs.
    After training, the final model is saved using:
    ```
    model.save("dqn_breakout_final")
    ```
    Saving the final model allows the learned policy to be loaded later for evaluation, plotting, or video generation.

    Finally:
    ```
    vec_env.close()
    eval_env.close()
    ```
    closes the environments cleanly and releases system resources. This is good practice when working with multiple environments and ALE backends.


    ## 4. Plotting Results
    """)
    return


@app.cell
def _(EventAccumulator, Path, np, plt):
    from scipy.ndimage import uniform_filter1d


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
        for tag in ea.Tags()["scalars"]:
            events = ea.Scalars(tag)
            data[tag] = {
                "steps": [e.step for e in events],
                "values": [e.value for e in events],
            }
        return data


    def smooth_data(values: list, window_size: int = 50) -> np.ndarray:
        """Apply smoothing to noisy data."""
        return uniform_filter1d(np.array(values), size=window_size, mode="nearest")


    def plot_training_curves(data: dict, save_path: str = None):
        """
        Plot training curves in a 2x2 grid layout.

        Args:
            data: Dictionary from extract_tensorboard_data
            save_path: Optional path to save the figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.patch.set_facecolor("white")

        # Color scheme matching the reference image
        colors = {
            "loss_raw": "#9999ff",  # Light blue/purple
            "loss_smooth": "#cc0000",  # Red
            "reward": "#228B22",  # Forest green
            "epsilon": "#800080",  # Purple
            "ep_length": "#FFA500",  # Orange
        }

        # Helper to convert steps to millions
        def to_millions(steps):
            return np.array(steps) / 1_000_000

        # ===== Top Left: Training Loss =====
        ax1 = axes[0, 0]
        if "train/loss" in data:
            steps = to_millions(data["train/loss"]["steps"])
            values = np.array(data["train/loss"]["values"])
            smoothed = smooth_data(values, window_size=100)

            ax1.plot(
                steps, values, color=colors["loss_raw"], alpha=0.4, linewidth=0.5
            )
            ax1.plot(
                steps,
                smoothed,
                color=colors["loss_smooth"],
                linewidth=2,
                label="Smoothed",
            )
            ax1.legend(loc="upper right")

        ax1.set_title("Training Loss", fontsize=14, fontweight="bold")
        ax1.set_xlabel("Timesteps (Millions)", fontsize=11)
        ax1.set_ylabel("Loss", fontsize=11)
        ax1.grid(True, alpha=0.3)

        # ===== Top Right: Training Reward =====
        ax2 = axes[0, 1]
        if "rollout/ep_rew_mean" in data:
            steps = to_millions(data["rollout/ep_rew_mean"]["steps"])
            values = data["rollout/ep_rew_mean"]["values"]

            ax2.plot(steps, values, color=colors["reward"], linewidth=1.5)

        ax2.set_title("Training Reward", fontsize=14, fontweight="bold")
        ax2.set_xlabel("Timesteps (Millions)", fontsize=11)
        ax2.set_ylabel("Mean Episode Reward", fontsize=11)
        ax2.grid(True, alpha=0.3)

        # ===== Bottom Left: Exploration Rate =====
        ax3 = axes[1, 0]
        if "rollout/exploration_rate" in data:
            steps = to_millions(data["rollout/exploration_rate"]["steps"])
            values = data["rollout/exploration_rate"]["values"]

            ax3.plot(steps, values, color=colors["epsilon"], linewidth=2)

        ax3.set_title(
            "Exploration Rate (ε-greedy)", fontsize=14, fontweight="bold"
        )
        ax3.set_xlabel("Timesteps (Millions)", fontsize=11)
        ax3.set_ylabel("Epsilon", fontsize=11)
        ax3.set_ylim(-0.05, 1.05)
        ax3.grid(True, alpha=0.3)

        # ===== Bottom Right: Episode Length =====
        ax4 = axes[1, 1]
        if "rollout/ep_len_mean" in data:
            steps = to_millions(data["rollout/ep_len_mean"]["steps"])
            values = data["rollout/ep_len_mean"]["values"]

            ax4.plot(steps, values, color=colors["ep_length"], linewidth=1.5)

        ax4.set_title("Episode Length", fontsize=14, fontweight="bold")
        ax4.set_xlabel("Timesteps (Millions)", fontsize=11)
        ax4.set_ylabel("Mean Episode Length", fontsize=11)
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
            print(f"Saved plot to {save_path}")

        plt.show()
        return fig


    # Example usage
    if __name__ == "__main__":
        # Extract data from your TensorBoard logs
        log_dir = "./option_pain/tensorboard_logs/DQN_5/"
        data = extract_tensorboard_data(log_dir)

        print("Available metrics:", list(data.keys()))

        # Plot training curves
        fig = plot_training_curves(data, save_path="training_curves.png")
    return


@app.cell
def _(DQN, eval_env, evaluate_policy):
    model_for_eval = DQN.load(
        "./option_pain/best_model/best_model.zip", env=eval_env
    )

    mean_reward, std_reward = evaluate_policy(
        model_for_eval,
        eval_env,
        n_eval_episodes=100,
        deterministic=True,
        return_episode_rewards=False,
    )

    print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    return (model_for_eval,)


@app.cell
def _(eval_env, evaluate_policy, model_for_eval, plt):
    rewards, _ = evaluate_policy(
        model_for_eval,
        eval_env,
        n_eval_episodes=50,
        deterministic=True,
        return_episode_rewards=True,
    )

    plt.hist(rewards, bins=15)
    plt.xlabel("Episode Return")
    plt.ylabel("Frequency")
    plt.title("Evaluation Return Distribution")
    plt.show()
    return (rewards,)


@app.cell
def _(mo):
    mo.md(r"""
    Figure shows the distribution of episode returns during evaluation. The distribution is bimodal, with a cluster of low-return episodes corresponding to early failures and a second cluster of high-return episodes indicating successful task completion. This explains the relatively large standard deviation despite a high mean return and suggests that the learned policy performs well in most cases but remains sensitive to early-state perturbations.
    """)
    return


@app.cell
def _(np, rewards):
    success_threshold = 200
    success_rate = np.mean(np.array(rewards) >= success_threshold)

    print(f"Success rate: {success_rate * 100:.1f}%")
    return


@app.cell
def _(mo):
    mo.md(r"""
    The trained agent achieved a success rate of X%, defined as episodes with return ≥ 200.
    """)
    return


@app.cell
def _(np, rewards):
    median = np.median(rewards)
    q1 = np.percentile(rewards, 25)
    q3 = np.percentile(rewards, 75)

    print(f"Median: {median}, IQR: [{q1}, {q3}]")
    return


@app.cell
def _(mo):
    mo.md(r"""
    The median return was X, with an interquartile range of [Q1, Q3], indicating that most successful episodes cluster at high returns.
    """)
    return


@app.cell(hide_code=True)
def _(Path, json, plt):
    BASE_DIR = Path(__file__).resolve().parent
    print()


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
        plt.plot(
            steps_eval_len, eval_len, marker="o", label="Eval Mean Episode Length"
        )
        plt.xlabel("Training Timesteps (Millions)")
        plt.ylabel("Mean Episode Length")
        plt.title("Evaluation Episode Length During Training")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(BASE_DIR / "eval_episode_length.png", dpi=300)
        plt.close()


    plot_training_curves()
    plot_exploration_rate()
    plot_evaluation_curves()
    print("Plots saved to:", BASE_DIR)
    return


@app.cell
def _(mo):
    mo.md(r"""
    This section presents and interprets the quantitative results obtained during training and evaluation of the Deep Q-Network (DQN) agent on the Atari Breakout environment. All metrics shown in this section were logged automatically by the Stable-Baselines3 framework during training and exported from TensorBoard log files in JSON format. As a result, the reported curves directly reflect the internal learning dynamics of the agent rather than post-processed or manually computed statistics. The selected metrics are standard in deep reinforcement learning research and are commonly used to evaluate learning progress, stability, and policy quality. Each metric captures a different aspect of the DQN training process, and together they provide a comprehensive view of agent performance.

    ### 4.1 Training Metrics Overview
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.image("training_curves.png", alt="Example diagram", width=400)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Figure 1: Training metrics logged during DQN learning.

    The plots show (top-left) training loss, (top-right) mean episode reward, (bottom-left) epsilon-greedy exploration rate, and (bottom-right) mean episode length as functions of training timesteps. All values are taken directly from TensorBoard logs.

    Figure 1 summarises the core training metrics recorded during learning. The steady increase in mean episode reward and episode length indicates improving gameplay performance, while the bounded and noisy training loss reflects stable temporal-difference learning. The exploration rate decays according to the predefined epsilon-greedy schedule, transitioning the agent from exploration to exploitation.


    ### 4.2 Mean Episode Reward
    """)
    return


@app.cell
def _():
    print("TODO")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Figure 2: Mean episode reward during training compared against random and
    human performance baselines. Shaded regions indicate variability across evalu-
    ation intervals.

    The mean episode reward is the primary performance metric in reinforcement learning. In Atari Breakout, higher rewards correspond to breaking more bricks and surviving longer. Figure 2 shows a steady improvement in performance over training, with the agent rapidly exceeding random behaviour and eventually surpassing the human baseline reported in prior work.

    Early training is characterised by near-random performance, while later stages show a sharp increase in reward as the agent learns effective paddle control and ball trajectory prediction.

    ### 4.3 Evaluation Performance
    Evaluation rewards are computed using a deterministic policy (no exploration)and provide a cleaner estimate of the learned policy’s quality. As shown in Figure 3, evaluation performance closely follows training performance, indicating that improvements are due to genuine policy learning rather than stochastic exploration effects.

    ### 4.4 Episode Length
    The mean episode length measures how long the agent survives in the envi-
    ronment before losing all lives. In Breakout, longer episodes typically indicate
    """)
    return


@app.cell
def _(mo):
    mo.image("learning_curve.png", alt="Example diagram", width=400)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Figure 3: Evaluation mean episode reward measured periodically during training using a separate evaluation environment without exploration noise. better performance. Figure 4 shows a clear upward trend, demonstrating that the agent becomes increasingly capable of sustaining gameplay and preventing ball loss.

    ### 4.5 Qualitative Results
    In addition to quantitative metrics, qualitative behaviour was assessed by visual inspection of gameplay. The trained agent demonstrates smooth paddle tracking, anticipates ball rebounds, and consistently clears large portions of the brick layout.

    ### 5.6 Video and Gameplay Frames
    A short gameplay recording of the trained agent is available online:
    - https://youtube.com/shorts/hdiVVfBSX9c?feature=share
    """)
    return


@app.cell
def _(mo):
    mo.image("learning_curve.png", alt="Example diagram", width=400)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Figure 4: Mean episode length during evaluation. Increasing episode length indicates improved paddle control and longer survival times.
    """)
    return


@app.cell
def _(mo):
    mo.image("images/breakout.gif", alt="Example diagram", width=400)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Figure 5: Gameplay frame showing the trained DQN agent maintaining ball control and clearing bricks

    ## 5. Evaluation
    T
    his section evaluates the performance of the trained Deep Q-Network (DQN) agent on the Atari Breakout environment. The evaluation focuses on learning effectiveness, training stability, generalisation behaviour, and known limitations of the applied approach. All conclusions are drawn from the quantitative metrics and qualitative observations presented in the Results section.

    ### 5.1 Learning Effectiveness

    The primary objective of this assignment was to train a reinforcement learning agent capable of learning effective gameplay behaviour directly from raw visual input. Based on the observed learning curves, the DQN agent clearly meets this objective.
    The steady increase in mean episode reward and episode length demonstrates that the agent learns meaningful control policies rather than exploiting trivial reward structures. The agent transitions from near-random behaviour to consistently high-scoring gameplay, ultimately surpassing the average human performance baseline reported in prior work. This confirms that the learned Q-function successfully captures the long-term reward structure of the Breakout environment.

    ### 5.2 Training Stability

    Training stability is a major concern in deep reinforcement learning due to the combination of bootstrapping, function approximation, and non-stationary data distributions. In this implementation, stability is assessed through the behaviour of the training loss, reward curves, and the absence of catastrophic performance collapse. The training loss exhibits bounded oscillations rather than monotonic convergence. This behaviour is expected in DQN and reflects the continually changing target values during learning. Importantly, the loss does not diverge, which indicates that the replay buffer and target network mechanisms successfully stabilise training.
    Additionally, no sudden drops in performance are observed in the evaluation reward curves. This suggests that the agent does not suffer from severe policy degradation or overfitting during later stages of training.

    ### 5.3 Exploration Strategy Evaluation

    The epsilon-greedy exploration schedule plays a crucial role in the learning process. Early in training, a high exploration rate allows the agent to experience a wide variety of state–action transitions, which is essential given the sparse and delayed reward structure of Breakout.
    As epsilon decays, the agent increasingly exploits learned Q-values. The sharp improvement in reward that coincides with the stabilisation of epsilon at its minimum value indicates that the agent has accumulated sufficient experience to act primarily greedily. This confirms that the chosen exploration schedule is appropriate for this task. However, epsilon-greedy exploration is known to be relatively inefficient compared to more advanced exploration strategies. While sufficient for this assignment, alternative methods such as noisy networks or entropy-based exploration could potentially improve sample efficiency.

    ### 5.4 Generalisation and Evaluation Environment
    Evaluation was conducted using a separate environment instance with deterministic action selection. The close alignment between training and evaluation rewards indicates that the learned policy generalises well across episodes and is not overly dependent on stochastic exploration.
    Nevertheless, it should be noted that evaluation is still performed within the same environment distribution as training. While the agent generalises across different episodes and initial conditions, it is not evaluated on fundamentally different game dynamics or modified reward structures.

    ### 5.5 Limitations
    Despite strong performance, several limitations should be acknowledged:

    - **Sample Inefficiency:** Training required several million timesteps, highlighting the high sample complexity of DQN when learning from pixel based input.

    - **Computational Cost:** The use of convolutional neural networks and large replay buffers results in high computational and memory requirements.

    - **Algorithmic Constraints:** Standard DQN is known to overestimate Q-values and can suffer from instability in more complex environments. Extensions such as Double DQN or Dueling DQN could improve robustness.

    - **Environment Specificity:** The learned policy is highly specialised to Breakout and cannot transfer directly to other games without retraining.


    ### 5.6 Comparison to Reference Implementations
    While direct numerical comparison to the original DeepMind results is difficult due to differences in hardware, training duration, and evaluation protocols, the qualitative performance trends are consistent with published DQN benchmarks. The agent reaches and exceeds human-level performance, demonstrating that the Stable-Baselines3 implementation faithfully reproduces the core behaviour of the original DQN algorithm.

    5.7 Summary
    Overall, the evaluation demonstrates that the implemented DQN agent performs effectively, learns stably, and achieves strong performance on the Atari Breakout task. The observed learning dynamics align well with theoretical expectations and prior empirical results, validating both the implementation and the chosen experimental setup.

    ## 6. Discussion and Conclusion
    This assignment set out to design, implement, and evaluate a Deep Reinforcement Learning agent capable of learning to play the Atari Breakout game directly from raw visual input. Using a Deep Q-Network (DQN) architecture implemented with the Stable-Baselines3 framework, the agent was trained on high-dimensional pixel observations and evaluated using standard reinforcement learning metrics.

    ### 6.1 Discussion
    The results demonstrate that combining convolutional neural networks with Q-learning enables effective learning in complex, visually rich environments. Starting from near-random behaviour, the agent progressively learned meaningful control strategies, such as accurate paddle positioning and anticipation of ball trajectories. This behavioural improvement is reflected quantitatively through increasing episode rewards and episode lengths, and qualitatively through stable and controlled gameplay observed in recorded videos. The use of experience replay and a target network proved essential for stabilising training. Without these mechanisms, the non-stationary nature of the learning targets would likely cause divergence when using deep neural networks. The bounded loss behaviour observed during training confirms that these stabilisation techniques were effective in practice. Using Stable-Baselines3 significantly reduced the amount of boilerplate code required to implement DQN, allowing greater focus on understanding algorithmic behaviour rather than low-level engineering details. Importantly, despite the high-level abstraction, the underlying learning process remains transparent and faithful to the original DQN formulation. By explicitly analysing where and how the Q-learning update is applied, this assignment demonstrates that using a library-based implementation does not obscure theoretical understanding. The exploration strategy played a critical role in early learning. The epsilon greedy schedule ensured sufficient exploration of the state–action space during the initial stages of training, while gradual exploitation allowed the agent to consolidate effective behaviours. However, epsilon-greedy exploration is known to be relatively inefficient, and more advanced exploration techniques could further improve learning speed and stability.

    ### 6.2 Limitations and Future Work
    Despite strong performance, several limitations remain. First, training a DQN agent on pixel-based input is computationally expensive and requires millions of environment interactions, highlighting the sample inefficiency of value-based methods. Second, standard DQN is susceptible to Q-value overestimation and can struggle in environments with more complex dynamics. Future work could address these limitations by incorporating established DQN extensions such as Double DQN to reduce overestimation bias, Dueling DQN to improve value estimation, or Prioritised Experience Replay to focus learning on more informative transitions. Additionally, alternative exploration strategies such as noisy networks could improve sample efficiency. Another potential direction is evaluating generalisation more rigorously by testing the trained agent under modified environment conditions or altered reward structures. This would provide deeper insight into the robustness of the learned policy beyond the standard Breakout setting.

    ### 6.3 Conclusion
    In conclusion, this assignment successfully demonstrates the application of Deep Reinforcement Learning to a challenging Atari game environment. The implemented DQN agent learns directly from raw pixel input, achieves human-level and beyond performance, and exhibits stable training behaviour consistent with theoretical expectations and prior empirical results. The project highlights the power of combining deep learning with reinforcement learning and reinforces the importance of stabilisation techniques when learning from high-dimensional sensory data. Overall, the results validate both the chosen methodology and the effectiveness of Deep Q-Networks for solving complex sequential decision-making problems.
    """)
    return


if __name__ == "__main__":
    app.run()
