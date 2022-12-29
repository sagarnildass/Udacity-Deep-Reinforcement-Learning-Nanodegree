import os
import time
from collections import deque

import torch
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


class MyEnvironment():
    def __init__(self, env, brain_name, state_type, n_frames=4):
        """Wrapper around supplied UnityEnvironment

        Arguments:
            env {UnityEnvironment} -- Unity evironment
            brain_name {str} -- unity brain name
            state_type {str} -- One of: discrete, continuous
        Keyword Arguments:
            n_frames {int} -- In the case of visual input, how many frames to stack (default: {4})
        """
        self.env = env
        self.brain_name = brain_name
        self.state_type = state_type
        self.n_frames = n_frames
        self.states = deque(maxlen=n_frames)

    def get_state(self, env_info):
        """Get state from environment info

        Arguments:
            env_info {[unityagents.brain.BrainInfo]} -- Environment information

        Returns:
            [np.array] -- State
        """

        if self.state_type == 'discrete':

            # Return the raw state space
            return env_info.vector_observations[0]

        elif self.state_type == 'continuous':

            # Get state (N, H, W, C)
            state = env_info.visual_observations[0]

            # Convert to (N, C, H, W)
            state = np.transpose(state, axes=(0, 3, 1, 2))

            # Add to running list of states
            while len(self.states) < self.n_frames:
                self.states.append(state)
            self.states.append(state)

            # Return (N,C,F,H,W)
            return np.transpose(np.array(self.states), axes=(1, 2, 0, 3, 4))

    def initialize(self, train_mode=True):
        """Initialize environment and return state

        Arguments:
            brain_name {str} -- unity brain name

        Returns:
            [array-like] -- Initial state of environment
        """

        env_info = self.env.reset(train_mode=train_mode)[self.brain_name]
        return self.get_state(env_info)

    def step(self, action):
        """Perform action in environment.

        Arguments:
            action {int} -- action ID to take in environment

        Returns:
            [tuple] -- tuple of type (state, rewards, done)
        """

        env_info = self.env.step(action)[self.brain_name]
        state = self.get_state(env_info)
        reward = env_info.rewards[0]
        done = env_info.local_done[0]
        return state, reward, done


def train(env, agent, state_type, brain_name=None,
          episodes=5000, max_steps=1000,
          eps_start=1.0, eps_end=0.001, eps_decay=0.97,
          thr_score=13.0
          ):
    """Train agent in the environment
    Arguments:
        env {UnityEnvironment} -- Unity Environment
        agent {object} -- Agent to traverse environment
        state_type {str} -- type of state space. Options: discrete|pixels

    Keyword Arguments:
        brain_name {str} -- brain name for Unity environment (default: {None})
        episodes {int} -- number of training episodes (default: {5000})
        max_steps {int} -- maximum number of timesteps per episode (default: {1000})
        eps_start {float} -- starting value of epsilon (default: {1.0})
        eps_end {float} -- minimum value of epsilon (default: {0.001})
        eps_decay {float} -- factor (per episode) used for decreasing epsilon (default: {0.97})
        thr_score {float} -- threshold score for the environment to be solved (default: {13.0})
    """

    # Scores for each episode
    scores = []

    # Last 100 scores
    scores_window = deque(maxlen=100)

    # Average scores & steps after each episode (within window)
    avg_scores = []

    # Best score so far
    best_avg_score = -np.inf

    # Get environment
    environment = MyEnvironment(env, brain_name, state_type)

    # Loop over episodes
    time_start = time.time()
    eps = eps_start
    for i in range(1, episodes + 1):
        state = environment.initialize()

        # Play an episode
        score = 0
        for _ in range(max_steps):
            action = agent.act(state, eps)
            next_state, reward, done = environment.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        eps = max(eps_end, eps_decay * eps)

        # Update book-keeping variables
        scores_window.append(score)
        scores.append(score)
        avg_score = np.mean(scores_window)
        avg_scores.append(avg_score)
        if avg_score > best_avg_score:
            best_avg_score = avg_score

        # Info for user every 100 episodes
        n_secs = int(time.time() - time_start)
        print(
            f'Episode {i:6}\t Score: {score:.2f}\t Avg: {avg_score:.2f}\t Best Avg: {best_avg_score:.2f} Epsilon {eps:.4f}\t Memory: {len(agent.memory):6}\t Seconds: {n_secs:4}')
        time_start = time.time()

        # Check if done
        if avg_score >= thr_score:
            print(f'\nEnvironment solved in {i:d} episodes!\tAverage Score: {avg_score:.2f}')

            # Save the weights
            torch.save(
                agent.q_local.state_dict(),
                'logs/weights_{}_{}_{}.pth'.format(
                    env.name,
                    agent.model_name,
                    'double' if agent.enable_double else 'single'
                )
            )

            # Create plot of scores vs. episode
            _, ax = plt.subplots(1, 1, figsize=(7, 5))
            sns.lineplot(range(len(scores)), scores, label='Score', ax=ax)
            sns.lineplot(range(len(avg_scores)), avg_scores, label='Avg Score', ax=ax)
            ax.set_xlabel('Episodes')
            ax.set_xlabel('Score')
            ax.set_title('Agent: {}-{}'.format('double' if agent.enable_double else 'single', agent.model_name))
            ax.legend()
            plt.savefig('./logs/scores_{}_{}_{}.png'.format(
                env.name,
                agent.model_name,
                'double' if agent.enable_double else 'single'
            ))

            break


def test(env, agent, state_type, brain_name, checkpoint):
    """Let pre-trained agent play in environment

    Arguments:
        env {UnityEnvironment} -- Unity Environment
        agent {object} -- Agent to traverse environment
        state_type {str} -- type of state space. Options: discrete|pixels
        brain_name {str} -- brain name for Unity environment (default: {None})
        checkpoint {str} -- filepath to load network weights
    """

    # Load trained model
    agent.q_local.load_state_dict(torch.load(checkpoint))

    # Get environment
    environment = MyEnvironment(env, brain_name, state_type)

    # Initialize & interact in environment
    state = environment.initialize(train_mode=False)
    for _ in range(600):

        # Get action & perform step
        action = agent.act(state)
        state, _, done = environment.step(action)
        if done:
            break

        # Prevent too fast rendering
        time.sleep(1 / 60.)