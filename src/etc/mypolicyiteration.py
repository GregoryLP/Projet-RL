import csv
import sys
import random
import math
import copy
import queue
import numpy as np

class PolicyIteration:
	def __init__(self, params):
		self.params = params
		self.memberships = params.cluster

	def evaluate_policy(self, ed):
		values = np.random.rand(len(ed.states))
		return values

	def improve_policy(self, ed):
		new_policy = {}
		for state in ed.states:
			new_policy[state] = np.random.dirichlet(np.ones(len(ed.actions)))
		return new_policy

	def update(self, ed):
		values = self.evaluate_policy(ed)
		new_policy = self.improve_policy(ed, values)
		ed.newaction = np.random.choice(ed.actions, p=new_policy[ed.app])
		ed.newapp = np.argmax(self.memberships[ed.newaction])
