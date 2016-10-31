import argparse
import atexit
import random
from PIL import Image
import gym
import numpy as np
from DQAgent import DQAgent
from Logger import Logger
from evaluation import evaluate


# Functions
def preprocess_observation(obs):
	image = Image.fromarray(obs, 'RGB').convert('L').resize((84, 110))  # Convert to gray-scale and resize according to PIL coordinates
	return np.asarray(image.getdata(), dtype=np.uint8).reshape(image.size[1], image.size[0])  # Convert to array and return


def get_next_state(current, obs):
	# Next state is composed by the last 3 images of the previous state and the new observation
	return np.append(current[1:], [obs], axis=0)


def exit_handler():
	global DQA
	DQA.quit()


# I/O
parser = argparse.ArgumentParser()
parser.add_argument('-t', '--train', action='store_true', help='train the agent')
parser.add_argument('-l', '--load', type=str, default=None, help='load the neural network weights from the given path')
parser.add_argument('-v', '--novideo', action='store_true', help='suppress video output (useful to train on headless servers)')
parser.add_argument('-d', '--debug', help='run in debug mode (no output files)', action='store_true')
parser.add_argument('-e', '--environment', type=str,
					help='name of the OpenAI Gym environment to use (default: MsPacman-v0)\n'
						 'DeepMind paper: MsPacman-v0, BeamRider-v0, Breakout-v0, Enduro-v0, Pong-v0, Qbert-v0, Seaquest-v0, SpaceInvaders-v0',
					default='MsPacman-v0')
parser.add_argument('--minibatch-size', type=int, default=32, help='number of transitions to train the DQN on')
parser.add_argument('--replay-memory-size', type=int, default=1e6, help='number of samples stored in the replay memory')
parser.add_argument('--target-network-update-freq', type=int, default=10e3, help='frequency (number of DQN updates) with which the target DQN is updated')
parser.add_argument('--avg-val-computation-freq', type=int, default=50e3, help='frequency (number of DQN updates) with which the average reward and Q value are computed')
parser.add_argument('--discount-factor', type=float, default=0.99, help='discount factor for the environment')
parser.add_argument('--update-freq', type=int, default=4, help='frequency (number of steps) with which to train the DQN')
parser.add_argument('--learning-rate', type=float, default=0.00025, help='learning rate for RMSprop')
parser.add_argument('--momentum', type=float, default=0.95, help='momentum for RMSprop')
parser.add_argument('--squared-momentum', type=float, default=0.95, help='squared momentum for RMSprop')
parser.add_argument('--min-squared-gradient', type=float, default=0.01, help='constant added to the denominator of RMSprop update')
parser.add_argument('--epsilon', type=float, default=1, help='initial exploration rate for the agent')
parser.add_argument('--min-epsilon', type=float, default=0.1, help='final exploration rate for the agent')
parser.add_argument('--epsilon-decrease', type=float, default=9e-7, help='rate at which to linearly decrease epsilon')
parser.add_argument('--replay-start-size', type=int, default=50e3, help='minimum number of transitions (with fully random policy) to store in the replay memory before starting training')
parser.add_argument('--initial-random-actions', type=int, default=30, help='number of random actions to be performed by the agent at the beginning of each episode')

parser.add_argument('--dropout', type=float, default=0., help='dropout rate for the DQN')
parser.add_argument('--max-episodes', type=int, default=np.inf, help='maximum number of episodes that the agent can experience before quitting')
parser.add_argument('--max-episode-length', type=int, default=np.inf, help='maximum number of steps in an episode')
parser.add_argument('--max-frames-number', type=int, default=50e6, help='maximum number of frames during the whole algorithm')
parser.add_argument('--test-freq', type=int, default=250000, help='frequency (number of frames) with which to test the agent\'s performance')
parser.add_argument('--validation-frames', type=int, default=135000, help='number of frames to test the model in table 3 of DeepMind paper')
parser.add_argument('--test-states', type=int, default=30, help='number of states on which to compute the average Q value')
args = parser.parse_args()

if args.debug:
	print '####################################################'\
		  'WARNING: debug flag is set, output will not be saved'\
		  '####################################################'

logger = Logger(debug=args.debug, append=args.environment)
atexit.register(exit_handler)  # Make sure to always save the model when exiting

# Variables
average_score_buffer = []
average_Q_buffer = []
test_states = []

# Setup
env = gym.make(args.environment)
network_input_shape = (4, 110, 84)  # Dimension ordering: 'th'
DQA = DQAgent(
	env.action_space.n,
	network_input_shape,
	replay_memory_size=args.replay_memory_size,
	minibatch_size=args.minibatch_size,
	learning_rate=args.learning_rate,
	momentum=args.momentum,
	squared_momentum=args.squared_momentum,
	min_squared_gradient=args.min_squared_gradient,
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
avg_val_csv = 'average_values_training.csv'
logger.to_csv(training_csv, 'length,score')
logger.to_csv(test_csv, 'length,score')
logger.to_csv(avg_val_csv, 'avg_score,avg_Q')

episode = 0
frame_counter = 0

# Main loop
while episode < args.max_episodes:
	# Start episode
	logger.log("Episode %d" % (episode))
	score = 0

	# Observe reward and initialize first state
	observation = preprocess_observation(env.reset())
	current_state = np.array([observation, observation, observation, observation])  # Initialize the first state with the same 4 images

	t = 0
	frame_counter += 1
	# Main episode loop
	while t < args.max_episode_length:
		if frame_counter > args.max_frames_number:
			DQA.quit()

		# Render the game if video output is not suppressed
		if not args.novideo:
			env.render()

		# Select an action (at the beginning of the episode, actions are random)
		action = DQA.get_action(np.asarray([current_state]))

		# Observe reward and next state
		observation, reward, done, info = env.step(action)
		observation = preprocess_observation(observation)
		next_state = get_next_state(current_state, observation)

		frame_counter += 1

        if args.train:
    		# Store transition in replay memory
    		clipped_reward = 1 if (reward >= 1) else (-1 if (reward <= -1) else reward)  # Clip the reward
    		DQA.add_experience(np.asarray([current_state]),
                               action,
                               clipped_reward,
                               np.asarray([next_state]),
                               done)

    		# Train the network (sample batches from replay memory, generate targets using DQN_target and update DQN)
    		if t % args.update_freq == 0 and len(DQA.experiences) >= args.replay_start_size:
    			DQA.train()
    			# Every C DQN updates, update DQN_target
    			if DQA.training_count % args.target_network_update_freq == 0 and DQA.training_count >= args.target_network_update_freq:
    				DQA.reset_target_network()
    			# Every avg_val_computation_freq DQN updates, log the average reward and Q value
    			if DQA.training_count % args.avg_val_computation_freq == 0 and DQA.training_count >= args.avg_val_computation_freq:
    				logger.to_csv(avg_val_csv, [np.mean(average_score_buffer), np.mean(average_Q_buffer)])
    				# Clear the lists
    				del average_score_buffer[:]
    				del average_Q_buffer[:]

    		# Linear epsilon annealing
    		if len(DQA.experiences) >= args.replay_start_size:
    			DQA.update_epsilon()

		# After transition, switch state
		current_state = next_state

		score += reward  # Keep track of score
		# Logging
		if done or t == args.max_episode_length - 1:
			logger.to_csv(training_csv, [t, score])  # Save episode data in the training csv
			logger.log("Length: %d; Score: %d\n" % (t + 1, score))
			break

		t += 1

		# TEST
		if args.train:
            if frame_counter % args.test_freq == 0
    			t_evaluation, score_evaluation = evaluate(DQA, args, logger)
    			logger.to_csv(test_csv, [t_evaluation, score_evaluation])  # Save episode data in the training csv

        	# Keep track of score and average maximum Q value on the test states in order to compute the average
        	if len(test_states) < args.test_states:
        		for _ in range(random.randint(1, 5)):
        			test_states.append(DQA.get_random_state())
        	else:
        		average_score_buffer.append(score)
        		average_Q_buffer.append(np.mean([DQA.get_max_q(state) for state in test_states]))

	episode += 1
	# End episode
