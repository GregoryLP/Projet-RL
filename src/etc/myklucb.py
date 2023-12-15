import csv
import sys
import random
import math
import copy
import queue
import numpy as np

class KLUCB:
	def __init__(self, params):
		self.params = params
		self.memberships = params.cluster
		self.epsilon = 0.1  

	def kl_divergence(self, p, q):
		# Calcul de la divergence de Kullback-Leibler
		return np.sum(p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q)))

	def select_action(self, ed):
		estimated_means = np.zeros(len(ed.actions))

		for a in ed.actions:
			estimated_means[a] = ed.rewards[a] / ed.counts[a]

		for a in ed.actions:
			q = 1.0 - self.epsilon 
			p = estimated_means[a]

			while q - p > self.epsilon:
				q = p
				p = estimated_means[a] + np.sqrt((2 * np.log(ed.time_step)) / ed.counts[a])

			estimated_means[a] = p

		chosen_action = np.argmax(estimated_means)
		return chosen_action

	def update(self, ed):
		ed.newaction = self.select_action(ed)
		ed.newapp = np.argmax(self.memberships[ed.newaction])