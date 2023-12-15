import csv
import sys
import random
import math
import copy
import queue
import numpy as np

class ANN():
	def __init__(self, params):
		self.params = params
		self.memberships = params.cluster
		self.learning_rate = 0.01 
		self.input_size = 4 
		self.hidden_size = 4  
		self.output_size = 100

		self.weights_input_hidden = np.random.rand(self.input_size, self.hidden_size)
		self.bias_hidden = np.zeros((1, self.hidden_size))
		self.weights_hidden_output = np.random.rand(self.hidden_size, self.output_size)
		self.bias_output = np.zeros((1, self.output_size))

	def sigmoid(self, x):
		return 1 / (1 + np.exp(-x))

	def softmax(self, x):
		exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
		return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

	def forward(self, state):
		# Propagation avant dans le réseau
		hidden_input = np.dot(state, self.weights_input_hidden) + self.bias_hidden
		hidden_output = self.sigmoid(hidden_input)

		output_input = np.dot(hidden_output, self.weights_hidden_output) + self.bias_output
		action_probabilities = self.softmax(output_input)

		return action_probabilities, hidden_output

	def backward(self, state, target, hidden_output):
		# Rétropropagation pour ajuster les poids
		error = target - state
		output_delta = error * (action_probabilities * (1 - action_probabilities))
		hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
		hidden_delta = hidden_error * (hidden_output * (1 - hidden_output))

		# Mise à jour des poids
		self.weights_hidden_output += np.dot(hidden_output.T, output_delta) * self.learning_rate
		self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * self.learning_rate
		self.weights_input_hidden += np.dot(state.T, hidden_delta) * self.learning_rate
		self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * self.learning_rate

	def update(self, ed):
		state = np.array([ed.feature1, ed.feature2, ed.feature3, ed.feature4]) 
		target = np.array([0])

		action_probabilities, hidden_output = self.forward(state)

		self.backward(state, target, hidden_output)

		ed.newaction = np.random.choice(ed.actions, p=action_probabilities[0])
		ed.newapp = np.argmax(self.memberships[ed.newaction])