import csv
import sys
import random
import math
import copy
import queue
import numpy as np

class Thompson:
	def __init__(self, params):
		self.params = params
		self.arm_count = self.params.edDict[0].actions
		self.memberships = params.cluster
		self.alpha = np.ones(self.arm_count)
		self.beta = np.ones(self.arm_count)

	def update(self, ed):
		ed.action = self.get_action(ed)
		ed.newapp = np.argmax(self.memberships[ed.action])

	def get_action(self, ed):
		self.alpha[ed.action] += 1 # increment action count
		self.beta[ed.action] += 1/self.alpha[ed.action] * (ed.dr_mean - self.beta[ed.action]) # inc. update rule
		theta = np.random.beta(self.alpha, self.beta)
		return theta.argmax()