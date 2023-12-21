import csv
import sys
import random
import math
import copy
import queue
import numpy as np

class Qlearning():
	def __init__(self, params):
		self.params = params
		self.memberships = params.cluster 

	def update(self, ed):
		ed.newaction = np.random.choice(ed.actions, p=ed.randompolicy[ed.app])
		ed.newapp = np.argmax(self.memberships[ed.newaction])

	def __init__(self, params):
		self.params = params
		self.memberships = params.cluster
		self.actions = params.actions
		self.states = params.states
		self.reward = params.reward

		# Initialize the matrix with Q-values
		init_data = [[float(self.reward) for _ in self.states]
				for _ in self.actions]
		self._qmatrix = pd.DataFrame(data=init_data,
                                    index=possible_actions,
                                    columns=possible_states)

        # Save the parameters
		self._learn_rate = learning_rate
		self._discount_factor = discount_factor

	def get_best_action(self, state):
		return self._qmatrix[[state]].idxmax().iloc[0]

	def update_model(self, state, action, reward, next_state):
		q_sa = self._qmatrix.ix[action, state]
		max_q_sa_next = self._qmatrix.ix[self.get_best_action(next_state), next_state]
		r = reward
		alpha = self._learn_rate
		gamma = self._discount_factor

		# Do the computation
		new_q_sa = q_sa + alpha * (r + gamma * max_q_sa_next - q_sa)
		self._qmatrix.set_value(action, state, new_q_sa)