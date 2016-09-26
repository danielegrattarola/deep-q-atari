from DQNetwork import DQNetwork
import random
import numpy as np


class DQAgent:
	def __init__(self,
				 actions,
				 network_input_shape,
				 replay_memory_size=1024,
				 minibatch_size = 32,
				 learning_rate=0.01,
				 discount_factor=0.9,
				 dropout_prob=0.1,
				 epsilon=1,
				 epsilon_decrease_rate=0.99,
				 load_path=None,
				 logger=None):

		# Parameters
		self.network_input_shape = network_input_shape
		self.actions = actions  # Size of the discreet action space
		self.learning_rate = learning_rate
		self.dropout_prob = dropout_prob
		self.load_path = load_path
		self.replay_memory_size = replay_memory_size
		self.minibatch_size = minibatch_size
		self.discount_factor = discount_factor  # Discount factor
		self.epsilon = epsilon  # Coefficient for epsilon-greedy exploration
		self.epsilon_decrease_rate = epsilon_decrease_rate  # (inverse) Rate at which to make epsilon smaller, as training improves the agent's performance; epsilon = epsilon * rate
		self.min_epsilon = 0.1  # Minimum epsilon value
		self.logger = logger

		# Experience replay variables
		self.experiences = []
		self.training_count = 0

		# Instantiate the deep Q-network
		self.DQN = DQNetwork(
			self.actions,
			self.network_input_shape,
			learning_rate=self.learning_rate,
			discount_factor=self.discount_factor,
			minibatch_size=self.minibatch_size,
			dropout_prob=self.dropout_prob,
			load_path=self.load_path,
			logger=self.logger
		)

	def get_action(self, state, testing=False, force_random=False):
		# Poll DQN for Q-values, return argmax with probability 1-epsilon
		if force_random or (random.random() < self.epsilon if not testing else 0.05):
			print 'Random action...'
			return random.randint(0, self.actions - 1)
		else:
			q_values = self.DQN.predict(state)
			print 'DQN action...'
			return np.argmax(q_values)

	def add_experience(self, source, action, reward, dest, final):
		# Add a tuple (source, action, reward, dest, final) to experiences
		self.experiences.append({'source': source, 'action': action, 'reward': reward, 'dest': dest, 'final': final})
		if not len(self.experiences) % 100:
			print "Gathered %d samples of %d" % (len(self.experiences), self.replay_memory_size)

	def sample_batch(self):
		# Pop batch_size random samples from experiences and return them as a batch 
		batch = []
		for i in range(self.minibatch_size):
			batch.append(self.experiences.pop(random.randrange(0, len(self.experiences))))
		return np.asarray(batch)

	def must_train(self):
		# Returns true if the number of samples in experiences is greater than the batch size
		return len(self.experiences) >= self.replay_memory_size

	def train(self, update_epsilon=True):
		# Sample a batch from experiences, train the DCN on it, update the epsilon-greedy coefficient
		self.training_count += 1
		if self.logger is not None:
			self.logger.log('Training session #%d - epsilon: %f' %(self.training_count, self.epsilon))
		batch = self.sample_batch()
		self.DQN.train(batch)  # Train the DCN
		if update_epsilon:
			self.epsilon -= self.epsilon_decrease_rate  if self.epsilon > self.min_epsilon else self.min_epsilon  # Decrease the probability of picking a random action to improve exploitation

	def reset(self):
		self.DQN = DQNetwork(
			self.actions,
			self.network_input_shape,
			learning_rate=self.learning_rate,
			discount_factor=self.discount_factor,
			minibatch_size=self.minibatch_size,
			dropout_prob=self.dropout_prob,
			load_path=self.load_path,
			logger=self.logger
		)

	def quit(self):
		# Save the DCN, quit
		self.DQN.save()
