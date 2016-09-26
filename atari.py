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
parser.add_argument('-l', '--load', type=str, required=False, default=None, help='load the neural network weights from the given path')
parser.add_argument('-e', '--environment', type=str, help='Name of the OpenAI Gym environment to use (default: MsPacman-v0)\n'
														  'DeepMind paper: MsPacman-v0, BeamRider-v0, Breakout-v0, Enduro-v0, Pong-v0, Qbert-v0, Seaquest-v0, SpaceInvaders-v0', required=False, default='MsPacman-v0')
parser.add_argument('-v', '--novideo', action='store_true', help='suppress video output (useful to train on headless servers)')
parser.add_argument('-d', '--debug', help='Run in debug mode (no output files)', action='store_true')

parser.add_argument('--replay-memory-size', type=int, required=False, default=1048576, help='')
parser.add_argument('--replay-start-size', type=int, required=False, default=50000, help='')
parser.add_argument('--minibatch-size', type=int, required=False, default=32, help='')

parser.add_argument('--learning-rate', type=float, required=False, default=0.00025, help='custom learning rate for the DQN (default: 0.00025)')
parser.add_argument('--discount-factor', type=float, required=False, default=0.99, help='custom discount factor for the environment (default: 0.99)')
parser.add_argument('--dropout', type=float, required=False, default=0.1, help='custom dropout rate for the DQN (default: 0.1)')
parser.add_argument('--epsilon', type=float, required=False, default=1, help='custom random exploration rate for the agent (default: 1)')
parser.add_argument('--epsilon-decrease', type=float, required=False, default=0.0000009, help='custom rate at which to linearly decrease epsilon (default: 0.001)')
parser.add_argument('--max-episodes', type=int, required=False, default=10000, help='maximum number of episodes that the agent can experience before quitting (default: 10000)')
parser.add_argument('--max-episode-length', type=int, required=False, default=10000, help='maximum number of steps in an episodes (default: 10000)')
parser.add_argument('--max-training-sessions', type=int, required=False, default=100, help='maximum number of training sessions before quitting (default: 1000)')
parser.add_argument('--initial-random-actions', type=int, required=False, default=30, help='')

parser.add_argument('--target-network-update-freq', type=int, required=False, default=10000, help='')
parser.add_argument('--test-freq', type=int, required=False, default=10, help='')

args = parser.parse_args()
if args.debug:
	print '####################################################' \
		  'WARNING: debug flag is set, output will not be saved' \
		  '####################################################'

logger = Logger(debug=args.debug, append=args.environment)
atexit.register(exit_handler)

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
	load_path=args.load,
	logger=logger
)

# Initial logging
logger.log({
	'Environment': args.environment,
	'Action space': env.action_space.n,
	'Observation space': env.observation_space.shape,
	'Learning rate': args.learning_rate,
	'Discount factor': args.discount_factor,
	'Dropout prob': args.dropout,
	'Epsilon': args.epsilon,
	'Epsilon decrease rate': args.epsilon_decrease,
	'Max episodes': args.max_episodes,
	'Max episode length': args.max_episode_length,
	'Max training sessions': args.max_training_sessions
})
training_csv = 'training_info.csv'
test_csv = 'test_info.csv'
logger.to_csv(training_csv, 'length,score')
logger.to_csv(test_csv, 'length,score')

# Main loop
for episode in range(args.max_episodes):
	# Quit if we reach the maximum number of training sessions allowed
	if DQA.training_count > args.max_training_sessions:
		break

	logger.log("Episode %d %s" % (episode, '(test)' if must_test else ''))
	score = 0
	remaining_random_actions = args.initial_random_actions
	observation = preprocess_observation(env.reset_target_network())
	current_state = np.array([observation, observation, observation, observation]) # Initialize the first state with the same 4 images

	for t in range(args.max_episode_length):
		if not args.novideo:
			env.render()

		# Select an action (at the beginning of the episode, actions are random)
		remaining_random_actions -= 1
		action = DQA.get_action(np.asarray([current_state]), testing=must_test, force_random=(remaining_random_actions >= 0))

		# Observe reward and next state
		observation, reward, done, info = env.step(action)
		observation = preprocess_observation(observation)
		next_state = get_next_state(current_state, observation)

		# Keep track of score
		score += reward

		# Store transition in replay memory
		if not must_test:
			clipped_reward = 1 if reward > 0 else (-1 if reward < 0 else 0) # Clip the reward like in the Deepmind paper
			DQA.add_experience(np.asarray([current_state]), action, clipped_reward, np.asarray([next_state]), done)

		# Train the network (sample batches from replay memory, generate targets using DQN_target and update DQN)
		if t % 4 == 0 and len(DQA.experiences) >= args.replay_start_size:
			DQA.train()

		# Every C step reset DQN_target
		if DQA.training_count >= args.target_network_update_freq:
			DQA.reset_target_network()

		# Logging
		if done or t == args.max_episode_length - 1:
			if must_test:
				logger.to_csv(test_csv, [t, score])
			else:
				logger.to_csv(training_csv, [t, score])
			logger.log("Length: %d; Score: %d\n" % (t + 1, score))
			must_test = (episode % args.test_freq == 0)


