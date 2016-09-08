from DQNetwork import DQNetwork
import random
import numpy as np


class DQAgent:
	def __init__(self,
				 actions,
				 batch_size=1024,
				 learning_rate=0.01,
				 discount_factor=0.9,
				 dropout_prob=0.1,
				 epsilon=1,
				 epsilon_decrease_rate=0.99,
				 network_input_shape=(4, 84, 84),
				 load_path='',
				 logger=None):

		self.actions = actions  # Size of the discreet action space
		# Training parameters
		self.batch_size = batch_size
		# Hyperparameters
		self.discount_factor = discount_factor  # Discount factor
		self.epsilon = epsilon  # Coefficient for epsilon-greedy exploration
		self.epsilon_decrease_rate = epsilon_decrease_rate  # (inverse) Rate at which to make epsilon smaller, as training improves the agent's performance; epsilon = epsilon * rate
		self.min_epsilon = 0.1  # Minimum epsilon value
		# Experience variables
		self.experiences = []
		self.training_count = 0

		# Instantiate the deep Q-network
		self.DQN = DQNetwork(
			self.actions,
			network_input_shape,
			gamma=self.discount_factor,
			dropout_prob=dropout_prob,
			load_path=load_path,
			logger=logger
		)

		if logger is not None:
			logger.log({
				'Discount factor': self.discount_factor,
				'Starting epsilon': self.epsilon,
				'Epsilon decrease rate': self.epsilon_decrease_rate,
				'Batch size': self.batch_size
			})

	def get_action(self, state, testing=False):
		# Poll DQN for Q-values, return argmax with probability 1-epsilon
		q_values = self.DQN.predict(state)
		if random.random() < self.epsilon and not testing:
			return random.randint(0, self.actions - 1)
		else:
			return np.argmax(q_values)

	def add_experience(self, source, action, reward, dest, final):
		# Add a tuple (source, action, reward, dest, final) to experiences
		self.experiences.append({'source': source, 'action': action, 'reward': reward, 'dest': dest, 'final': final})
		if not len(self.experiences) % 100:
			print "Gathered %d samples of %d" % (len(self.experiences), self.batch_size)

	def sample_batch(self):
		# Pop batch_size random samples from experiences and return them as a batch 
		batch = []
		for i in range(self.batch_size):
			batch.append(self.experiences.pop(random.randrange(0, len(self.experiences))))
		return np.asarray(batch)

	def must_train(self):
		# Returns true if the number of samples in experiences is greater than the batch size
		return len(self.experiences) >= self.batch_size

	def train(self, update_epsilon=True):
		# Sample a batch from experiences, train the DCN on it, [optionally] update the epsilon-greedy coefficient
		self.training_count += 1
		print 'Training session #', self.training_count, ' - epsilon:', self.epsilon
		batch = self.sample_batch()
		self.DQN.train(batch)  # Train the DCN
		if update_epsilon:
			self.epsilon = self.epsilon * self.epsilon_decrease_rate if self.epsilon > self.min_epsilon else self.min_epsilon  # Decrease the probability of picking a random action to improve exploitation

	def quit(self):
		# Stop experiencing episodes, save the DCN, quit
		self.DQN.save()
