from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
import numpy as np


class DQNetwork:
	def __init__(self, actions, input_shape, learning_rate=0.01, discount_factor=0.99, dropout_prob=0.1, load_path=None, logger=None):
		self.model = Sequential()
		self.actions = actions  # Size of the network output
		self.discount_factor = discount_factor
		self.learning_rate = learning_rate
		self.dropout_prob = dropout_prob

		# Deep Q Network as defined in the DeepMind paper
		# Ordering th: (samples, channels, rows, cols)
		self.model.add(Convolution2D(16, 8, 8, border_mode='valid', subsample=(4, 4), input_shape=input_shape))
		self.model.add(Activation('relu'))

		self.model.add(Convolution2D(32, 4, 4, border_mode='valid', subsample=(2, 2)))
		self.model.add(Activation('relu'))

		self.model.add(Flatten())

		self.model.add(Dense(256))
		self.model.add(Activation('relu'))

		self.model.add(Dense(self.actions))

		self.optimizer = RMSprop(lr=self.learning_rate)
		self.logger = logger

		# Load the network from saved model
		if load_path is not None:
			self.load(load_path)

		self.model.compile(loss='mean_squared_error', optimizer=self.optimizer, metrics=['accuracy'])

	def train(self, batch):
		# Generate the xs and targets for the given batch, train the model on them
		# The batch must be composed of SARS tuples as python dictionaries with labels 'source', 'action', 'dest', 'reward'
		x_train = []
		t_train = []

		# Generate training set and targets
		for datapoint in batch:
			x_train.append(datapoint['source'])

			# Get the current Q-values for the next state and select the best
			next_state_pred = list(self.predict(datapoint['dest']))
			next_a_Q_value = np.max(next_state_pred)

			# Set the target so that error will be 0 on all actions except the one taken
			t = list(self.predict(datapoint['source'])[0])
			t[datapoint['action']] = (datapoint['reward'] + self.discount_factor * next_a_Q_value) if not datapoint['final'] else \
			datapoint['reward']

			t_train.append(t)

		print next_state_pred  # Print a prediction so to have an idea of the Q-values magnitude
		x_train = np.asarray(x_train).squeeze()
		t_train = np.asarray(t_train).squeeze()
		history = self.model.fit(x_train, t_train, batch_size=32, nb_epoch=1)
		self.logger.to_csv('training_history.csv', [history.history['loss'], history.history['acc']])

	def predict(self, state):
		# Feed state into the model, return predicted Q-values
		return self.model.predict(state, batch_size=1)

	def save(self, filename=None):
		# Save the model and its weights to disk
		print 'Saving...'
		self.model.save_weights(self.logger.path + ('model.h5' if filename is None else filename))

	def load(self, path):
		# Load the model and its weights from path
		print 'Loading...'
		self.model.load_weights(path)
