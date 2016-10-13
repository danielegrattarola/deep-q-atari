from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
import numpy as np


class DQNetwork:
	def __init__(self, actions, input_shape, minibatch_size=32, learning_rate=0.01, discount_factor=0.99, dropout_prob=0.1, load_path=None, logger=None):
		self.model = Sequential()
		self.actions = actions  # Size of the network output
		self.discount_factor = discount_factor
		self.minibatch_size = minibatch_size
		self.learning_rate = learning_rate
		self.dropout_prob = dropout_prob
		self.logger = logger
		self.training_history_csv = 'training_history.csv'
		if self.logger is not None:
			self.logger.to_csv(self.training_history_csv, 'Loss,Accuracy')

		# Deep Q Network as defined in the DeepMind article on Nature
		# Ordering th: (samples, channels, rows, cols)

		# First convolutional layer
		self.model.add(Convolution2D(32, 8, 8, border_mode='valid', subsample=(4, 4), input_shape=input_shape, dim_ordering='th'))
		self.model.add(Activation('relu'))

		# Second convolutional layer
		self.model.add(Convolution2D(64, 4, 4, border_mode='valid', subsample=(2, 2), dim_ordering='th'))
		self.model.add(Activation('relu'))

		# Third convolutional layer
		self.model.add(Convolution2D(64, 3, 3, border_mode='valid', subsample=(1, 1), dim_ordering='th'))
		self.model.add(Activation('relu'))

		self.model.add(Flatten())

		# First dense layer
		self.model.add(Dense(512))
		self.model.add(Activation('relu'))

		# Output layer
		self.model.add(Dense(self.actions))

		# Optimization algorithm
		self.optimizer = Adam()

		# Load the network weights from saved model
		if load_path is not None:
			self.load(load_path)

		self.model.compile(loss='mean_absolute_error', optimizer=self.optimizer, metrics=['accuracy'])

	def train(self, batch, DQN_target):
		# Generate the xs and targets for the given batch, train the model on them
		# The batch must be composed of SARS tuples as python dictionaries with labels 'source', 'action', 'dest', 'reward'
		x_train = []
		t_train = []

		# Generate training set and targets
		for datapoint in batch:
			x_train.append(datapoint['source'].astype(np.float64))

			# Get the Q-values for the next state from the target DQN and select the best of them
			next_state_pred = DQN_target.predict(datapoint['dest'].astype(np.float64)).ravel()
			next_Q_value = np.max(next_state_pred)

			# Set the target so that error will be 0 on all actions except the one taken
			t = list(self.predict(datapoint['source'])[0])
			t[datapoint['action']] = (datapoint['reward'] + self.discount_factor * next_Q_value) if not datapoint['final'] else datapoint['reward']
			t[datapoint['action']] = 1 if (t[datapoint['action']] >= 1) else (-1 if (t[datapoint['action']] <= -1) else t[datapoint['action']])  # Clip the target

			t_train.append(t)

		# print next_state_pred  # Print a prediction so to have an idea of the Q-values magnitude
		x_train = np.asarray(x_train).squeeze()
		t_train = np.asarray(t_train).squeeze()
		history = self.model.fit(x_train, t_train, batch_size=self.minibatch_size, nb_epoch=1)
		if self.logger is not None:
			self.logger.to_csv(self.training_history_csv, [history.history['loss'][0], history.history['acc'][0]])

	def predict(self, state):
		# Feed state into the DQN, return predicted Q-values
		state = state.astype(np.float64)
		return self.model.predict(state, batch_size=1)

	def save(self, filename=None, append=''):
		# Save the DQN weights to disk
		f = (('model%s.h5' % append) if filename is None else filename)
		if self.logger is not None:
			self.logger.log('Saving model as %s' % f)
		self.model.save_weights(self.logger.path + f)

	def load(self, path):
		# Load the model and its weights from path
		if self.logger is not None:
			self.logger.log('Loading weights from file...')
		self.model.load_weights(path)
