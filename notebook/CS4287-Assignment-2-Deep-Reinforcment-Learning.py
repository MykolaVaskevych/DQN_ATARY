import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    **Team Members:**  Mykola Vaskevych (22372199) , Oliver Fitzgerald (22365958)
    **Execution:** Code Does **NOT** execute to the end without error
    **Third Party Implmentations Used:** TBD
    """)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # CS4287 Assignment 2 Deep Reinforcment Learning

    In this project we set out with the objective to train a deep reinforcment learning model to learn to play the game [breakout](https://ale.farama.org/environments/breakout/) from OpenAis Gymnasium to a comperable level to a human.

    ## 1. Introduction

    To accomplish this objective our agent must learn directly from raw visual input (images) rather than low-dimensional state variables (raw numbers). This makes the task more challenging as it will require the model to process images and extract features it needs as well as the actions to take on the perceived state. Rather than taking actions on a definitivly defined state such as in a classical game like chess where the board state is definite i.e there is no room for interperataion by the model.

    ### 1.1 Our Approach

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

    #### **Why We Chose Deep Q-Network**

    TODO
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
