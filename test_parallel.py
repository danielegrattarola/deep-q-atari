import gym
from DQAgent import DQAgent
import evaluation

env = gym.make("Breakout-v0")
network_input_shape = (4, 110, 84)  # Dimension ordering: 'th'
DQA = DQAgent(
    env.action_space.n,
    network_input_shape
)

episodes = evaluation.collectEpisode(env, DQA)

