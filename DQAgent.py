from DQNetwork import DQNetwork
import random
import numpy as np


class DQAgent:
	def __init__(self,
				 actions,
				 network_input_shape,
				 replay_memory_size=1024,
				 minibatch_size=32,
				 learning_rate=0.00025,
				 momentum=0.95,
				 squared_momentum=0.95,
				 min_squared_gradient=0.01,
				 discount_factor=0.9,
				 dropout_prob=0.1,
				 epsilon=1,
				 epsilon_decrease_rate=0.99,
				 min_epsilon=0.1,
				 load_path=None,
				 logger=None):

		# Parameters
		self.network_input_shape = network_input_shape
		self.actions = actions
		self.learning_rate = learning_rate
		self.momentum = momentum
		self.squared_momentum = squared_momentum
		self.min_squared_gradient = min_squared_gradient
		self.dropout_prob = dropout_prob
		self.load_path = load_path
		self.replay_memory_size = replay_memory_size
		self.minibatch_size = minibatch_size
		self.discount_factor = discount_factor
		self.epsilon = epsilon
		self.epsilon_decrease_rate = epsilon_decrease_rate
		self.min_epsilon = min_epsilon  # Minimum epsilon value
		self.logger = logger

		# Replay memory
		self.experiences = []
		self.training_count = 0

		# Instantiate the deep Q-networks
		# Main DQN
		self.DQN = DQNetwork(
			self.actions,
			self.network_input_shape,
			learning_rate=self.learning_rate,
			momentum=self.momentum,
			squared_momentum=self.squared_momentum,
			min_squared_gradient=self.min_squared_gradient,
			discount_factor=self.discount_factor,
			minibatch_size=self.minibatch_size,
			dropout_prob=self.dropout_prob,
			load_path=self.load_path,
			logger=self.logger
		)
		# Target DQN used to generate targets
		self.DQN_target = DQNetwork(
			self.actions,
			self.network_input_shape,
			learning_rate=self.learning_rate,
			momentum=self.momentum,
			squared_momentum=self.squared_momentum,
			min_squared_gradient=self.min_squared_gradient,
			discount_factor=self.discount_factor,
			minibatch_size=self.minibatch_size,
			dropout_prob=self.dropout_prob,
			load_path=self.load_path,
			logger=self.logger
		)
		# Reset target DQN
		self.DQN_target.model.set_weights(self.DQN.model.get_weights())

	def get_action(self, state, testing=False, force_random=False):
		# Poll DQN for Q-values
		# Return argmax with probability 1-epsilon during training, 0.05 during testing
		if force_random or (random.random() < (self.epsilon if not testing else 0.05)):
			return random.randint(0, self.actions - 1)
		else:
			q_values = self.DQN.predict(state)
			return np.argmax(q_values)

	def get_max_q(self, state):
		# Returns the maximum Q value predicted on the given state
		q_values = self.DQN.predict(state)
		idxs = np.argwhere(q_values == np.max(q_values)).ravel()
		return np.random.choice(idxs)

	def get_random_state(self):
		return self.experiences[random.randrange(0, len(self.experiences))]['source']

	def add_experience(self, source, action, reward, dest, final):
		# Remove older transitions if the replay memory is full
		if len(self.experiences) >= self.replay_memory_size:
			self.experiences.pop(0)
		# Add a tuple (source, action, reward, dest, final) to replay memory
		self.experiences.append({'source': source, 'action': action, 'reward': reward, 'dest': dest, 'final': final})
		# Periodically log how many samples we've gathered so far
		if (len(self.experiences) % 100 == 0) and (len(self.experiences) < self.replay_memory_size) and (self.logger is not None):
			self.logger.log("Gathered %d samples of %d" % (len(self.experiences), self.replay_memory_size))

	def sample_batch(self):
		# Sample minibatch_size random transitions from experiences and return them as a batch
		batch = []
		for i in xrange(self.minibatch_size):
			batch.append(self.experiences[random.randrange(0, len(self.experiences))])
		return np.asarray(batch)

	def train(self):
		# Sample a batch from experiences and train the DQN on it
		self.training_count += 1
		print 'Training session #%d - epsilon: %f' % (self.training_count, self.epsilon)
		batch = self.sample_batch()
		self.DQN.train(batch, self.DQN_target)  # Train the DQN

	def update_epsilon(self):
		# Decrease the probability of picking a random action to improve exploitation
		self.epsilon = (self.epsilon - self.epsilon_decrease_rate) if (self.epsilon > self.min_epsilon) else self.min_epsilon

	def reset_target_network(self):
		# Update the target DQN with the current weights of the main DQN
		if self.logger is not None:
			self.logger.log('Updating target network...')
		self.DQN_target.model.set_weights(self.DQN.model.get_weights())

	def quit(self):
        if self.load_path is None:
    		# Save the DQN, quit
    		if self.logger is not None:
    			self.logger.log('Quitting...')
    		self.DQN.save(append='_DQN')
    		self.DQN_target.save(append='_DQN_target')
