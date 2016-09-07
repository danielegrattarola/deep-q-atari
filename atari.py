import gym
import numpy as np
from DQAgent import DQAgent
from DQNetwork import DQNetwork
from Logger import Logger
import argparse

# I/O
parser = argparse.ArgumentParser()
parser.add_argument('-e', '--environment', type=str, help='Name of the OpenAI Gym environment to use', required=False,
					default='MsPacman-v0')
parser.add_argument('-d', '--debug', help='Run in debug mode (no output files)', action='store_true')
args = parser.parse_args()
logger = Logger(debug=args.debug)

# Parameters

# Entities
env = gym.make(args.environment)

# Initial logging
logger.log({
	'Environment': args.environment,
	'Action space': env.action_space.n,
	'Observation space': env.observation_space.shape
})

# Main loop
source_state = np.zeros(env.observation_space.shape)

for i_episode in range(20):
	observation = env.reset()
	for t in range(1000):
		env.render()
		# print(observation)
		action = env.action_space.sample()
		observation, reward, done, info = env.step(action)
		if done:
			print("Episode finished after {} timesteps".format(t + 1))
			break
