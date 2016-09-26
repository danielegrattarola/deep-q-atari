import argparse
import atexit
from PIL import Image
import gym
import numpy as np
from DQAgent import DQAgent
from Logger import Logger


# Constants
IMAGE_SHAPE = (110, 84)

# Functions
def preprocess_observation(obs):
	image = Image.fromarray(obs, 'RGB').convert('L').resize(IMAGE_SHAPE) # Convert to grayscale and resize
	return np.asarray(image.getdata(), dtype=np.uint8).reshape(image.size[0], image.size[1]) # Convert to array and return

def get_next_state(current, obs):
	return np.append(current[1:], [obs], axis=0) # Next state is composed by the last 3 images of the previous state and the new observation

def exit_handler():
	global DQA
	DQA.quit()

# I/O
parser = argparse.ArgumentParser()
parser.add_argument('-t', '--train', action='store_true', help='train the agent')
parser.add_argument('-l', '--load', type=str, default=None, help='load the neural network weights from the given path')
parser.add_argument('-v', '--novideo', action='store_true', help='suppress video output (useful to train on headless servers)')
parser.add_argument('-d', '--debug', help='Run in debug mode (no output files)', action='store_true')
parser.add_argument('-e', '--environment', type=str,
					help='Name of the OpenAI Gym environment to use (default: MsPacman-v0)\n'
						 'DeepMind paper: MsPacman-v0, BeamRider-v0, Breakout-v0, Enduro-v0, Pong-v0, Qbert-v0, Seaquest-v0, SpaceInvaders-v0',
					default='MsPacman-v0')
parser.add_argument('--minibatch-size', type=int, default=32, help='number of transitions to train the DQN on')
parser.add_argument('--replay-memory-size', type=int, default=100000, help='number of samples stored in the replay memory')
parser.add_argument('--target-network-update-freq', type=int, default=10000, help='frequency (number of DQN updates) with which the target DQN is updated')
parser.add_argument('--discount-factor', type=float, default=0.99, help='discount factor for the environment')
parser.add_argument('--update-freq', type=int, default=4, help='frequency (number of steps) with which to train the DQN')
parser.add_argument('--learning-rate', type=float, default=0.00025, help='learning rate for the DQN')
# Missing: gradient momentum
# Missing: squared gradient momentum
# Missing: min squared gradient
parser.add_argument('--epsilon', type=float, default=1, help='initial exploration rate for the agent')
parser.add_argument('--min-epsilon', type=float, default=0.1, help='final exploration rate for the agent')
parser.add_argument('--epsilon-decrease', type=float, default=0.0000009, help='rate at which to linearly decrease epsilon')
parser.add_argument('--replay-start-size', type=int, default=5000, help='minimum number of transitions (with fully random policy) to store in the replay memory before starting training')
parser.add_argument('--initial-random-actions', type=int, default=30, help='number of random actions to be performed by the agent at the beginning of each episode')

parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate for the DQN')
parser.add_argument('--max-episodes', type=int, default=10000, help='maximum number of episodes that the agent can experience before quitting')
parser.add_argument('--max-episode-length', type=int, default=10000, help='maximum number of steps in an episode')
parser.add_argument('--test-freq', type=int, default=10, help='frequency (number of episodes) with which to test the agent\'s performance')

args = parser.parse_args()
if args.debug:
	print '####################################################' \
		  'WARNING: debug flag is set, output will not be saved' \
		  '####################################################'

logger = Logger(debug=args.debug, append=args.environment)
atexit.register(exit_handler) # Make sure to always save the model when exiting

# Variables
must_test = False

# Setup
env = gym.make(args.environment)
network_input_shape = (4,110,84) # Dimension ordering: 'th'
DQA = DQAgent(
	env.action_space.n,
	network_input_shape,
	replay_memory_size=args.replay_memory_size,
	learning_rate=args.learning_rate,
	discount_factor=args.discount_factor,
	dropout_prob=args.dropout,
	epsilon=args.epsilon,
	epsilon_decrease_rate=args.epsilon_decrease,
	min_epsilon=args.min_epsilon,
	load_path=args.load,
	logger=logger
)

# Initial logging
logger.log({
	'Action space': env.action_space.n,
	'Observation space': env.observation_space.shape
})
logger.log(vars(args))
training_csv = 'training_info.csv'
test_csv = 'test_info.csv'
logger.to_csv(training_csv, 'length,score')
logger.to_csv(test_csv, 'length,score')

# Main loop
for episode in range(args.max_episodes):

	logger.log("Episode %d %s" % (episode, '(test)' if must_test else ''))
	score = 0
	remaining_random_actions = args.initial_random_actions # The first actions are forced to be random

	# Observe reward and initialize first state
	observation = preprocess_observation(env.reset())
	current_state = np.array([observation, observation, observation, observation]) # Initialize the first state with the same 4 images

	for t in range(args.max_episode_length):
		# Render the game if video output is not suppressed
		if not args.novideo:
			env.render()

		# Select an action (at the beginning of the episode, actions are random)
		remaining_random_actions = (remaining_random_actions - 1) if remaining_random_actions >= 0 else -1 # Clipped to -1 to avoid overflow
		action = DQA.get_action(np.asarray([current_state]), testing=must_test, force_random=(remaining_random_actions >= 0))

		# Observe reward and next state
		observation, reward, done, info = env.step(action)
		observation = preprocess_observation(observation)
		next_state = get_next_state(current_state, observation)

		if not must_test:
			# Store transition in replay memory
			clipped_reward = 1 if reward > 0 else (-1 if reward < 0 else 0) # Clip the reward like in the Deepmind paper
			DQA.add_experience(np.asarray([current_state]), action, clipped_reward, np.asarray([next_state]), done)

			# Train the network (sample batches from replay memory, generate targets using DQN_target and update DQN)
			if t % args.update_freq == 0 and len(DQA.experiences) >= args.replay_start_size:
				DQA.train()

			# Every C network updates, update DQN_target
			if DQA.training_count % args.target_network_update_freq == 0 and DQA.training_count >= args.target_network_update_freq:
				DQA.reset_target_network()

			# Linear epsilon annealing
			if len(DQA.experiences) >= args.replay_start_size:
				DQA.update_epsilon()

		# Logging
		score += reward # Keep track of score
		if done or t == args.max_episode_length - 1:
			if must_test:
				logger.to_csv(test_csv, [t, score]) # Save episode data in the test csv
			else:
				logger.to_csv(training_csv, [t, score]) # Save episode data in the training csv
			logger.log("Length: %d; Score: %d\n" % (t + 1, score))
			must_test = (episode % args.test_freq == 0) # Every test_freq episodes we have a test episode
			break


