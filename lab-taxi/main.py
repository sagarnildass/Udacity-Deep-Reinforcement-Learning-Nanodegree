from agent import Agent
from monitor import interact
import gym
import numpy as np

alpha = 0.01
env = gym.make('Taxi-v3')
agent = Agent()
avg_rewards, best_avg_reward = interact(env, agent)
