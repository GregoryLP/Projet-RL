import csv
import sys
import random
import math
import copy
import queue
import numpy as np

class EpsilonGreedy():
	def __init__(self, params):
		self.params = params
		self.memberships = params.cluster 
		self.arm_count   = self.params.edDict[0].actions
		self.epsilon     = 0.4
		self.Q           = np.zeros(self.arm_count) # q-value of actions
		self.N           = np.zeros(self.arm_count) # action count

	def update(self, ed):
		ed.newaction = self.get_action(ed)
		ed.newapp = np.argmax(self.memberships[ed.newaction])

	def get_action(self, ed):
		self.N[ed.action] += 1 # increment action count
		self.Q[ed.action] += 1/self.N[ed.action] * (ed.dr_mean - self.Q[ed.action]) # inc. update rule

		if np.random.uniform(0,1) > self.epsilon:
			action = self.Q.argmax()
		else:
			action = np.random.randint(0, self.arm_count)
		return action
