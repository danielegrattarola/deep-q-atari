import gym
import numpy as np
import sys

from DQAgent import DQAgent
from Logger import Logger
import argparse
from PIL import Image

# Constants
IMAGE_SHAPE = (110, 84)
MAX_EPISODES = 1000
MAX_EPISODE_LENGTH = 10000
MAX_TRAINING_SESSIONS = 100

# Functions
def preprocess_observation(obs):
	image = Image.fromarray(obs, 'RGB').convert('L').resize(IMAGE_SHAPE) # Convert to greyscale and resize
	return np.asarray(image.getdata(), dtype=np.uint8).reshape(image.size[0], image.size[1]) # Convert to array and return

def get_next_state(current, obs):
	return np.append(current[1:], [obs], axis=0) # Next state is composed by the last 3 images of the previous state and the new observation


# I/O
parser = argparse.ArgumentParser()
parser.add_argument('-t', '--train', action='store_true', help='train the agent.')
parser.add_argument('-l', '--load', type=str, required=False, default=None, help='load the neural network from disk.')
parser.add_argument('-e', '--environment', type=str, help='Name of the OpenAI Gym environment to use', required=False, default='MsPacman-v0')
parser.add_argument('-v', '--novideo', action='store_true', help='suppress video output (useful to train on headless servers).')
parser.add_argument('-d', '--debug', help='Run in debug mode (no output files)', action='store_true')
parser.add_argument('--gamma', type=float, required=False, default=0.95, help='custom discount factor for the environment.')
parser.add_argument('--dropout', type=float, required=False, default=0.1, help='custom dropout rate for the Q-network.')
args = parser.parse_args()
logger = Logger(debug=args.debug)

# Variables
must_test = False

# Entities
env = gym.make(args.environment)
DQA = DQAgent(
	env.action_space.n,
	network_input_shape=(4,110,84),
	dropout_prob=args.dropout,
	load_path=args.load,
	logger=logger
)

# Initial logging
logger.log({
	'Environment': args.environment,
	'Action space': env.action_space.n,
	'Observation space': env.observation_space.shape
})
training_csv = 'training_data.csv'
test_csv = 'test_data.csv'
logger.to_csv(training_csv, 'episode,length,score')

# Main loop
for episode in range(MAX_EPISODES):
	# Quit if we reach the maximum number of training sessions allowed
	if DQA.training_count > MAX_TRAINING_SESSIONS:
		DQA.quit()
		sys.exit(0)

	logger.log("Episode %d" % episode)
	score = 0
	observation = preprocess_observation(env.reset())
	current_state = np.array([observation, observation, observation, observation]) # Initialize the first state with the same 4 images

	for t in range(MAX_EPISODE_LENGTH):
		if not args.novideo:
			env.render()

		action = DQA.get_action(np.asarray([current_state]), testing=must_test)
		observation, reward, done, info = env.step(action)
		observation = preprocess_observation(observation)
		score += reward
		next_state = get_next_state(current_state, observation)

		if not must_test:
			DQA.add_experience(np.asarray([current_state]), action, reward, np.asarray([next_state]), done)

		if done:
			must_test = False
			logger.log("Length: %d; Score: %d\n" % (t + 1, score))
			if DQA.must_train():
				DQA.train()
				must_test = True # Test the agent's skills after every training session
			logger.to_csv(training_csv if not must_test else test_csv, [episode, t, score])
			break
