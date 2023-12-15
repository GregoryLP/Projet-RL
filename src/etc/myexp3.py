import csv
import sys
import random
import math
import copy
import queue
import numpy as np

class EXP3():
	def __init__(self, params):
		self.params = params
		self.memberships = params['cluster'] 
		self.arm_count   = self.params.edDict[0].actions
		self.epsilon     = 0.4
		self.Q           = np.zeros(self.arm_count) # q-value of actions
		self.N           = np.zeros(self.arm_count) # action count

	def select_action(self, ed):
		ed.newaction = self.get_action(ed)
		ed.newapp = np.argmax(self.memberships[ed.newaction])

	def update_weights(self, chosen_action, reward, ed):
		estimated_reward = reward / action_probabilities[chosen_action]
		self.weights[chosen_action] *= np.exp(self.gamma * estimated_reward / len(ed.actions))

	def update(self, ed):
		ed.newaction = self.select_action(ed)
		ed.newapp = np.argmax(self.memberships[ed.newaction])

class EnvironmentData:
	def __init__(self, actions, app, randompolicy):
		self.actions = actions
		self.app = app
		self.randompolicy = randompolicy
		self.newaction = None
		self.newapp = None

params = {'cluster': np.array([[0.2, 0.8], [0.6, 0.4]])}
exp3_instance = EXP3(params)

actions = [1, 2, 3]
app = 0
randompolicy = {0: [0.3, 0.3, 0.4], 1: [0.2, 0.5, 0.3]}
env_data = EnvironmentData(actions, app, randompolicy)

exp3_instance.update(env_data)

