import csv
import sys
import random
import math
import copy
import queue
import numpy as np

class UCB:
    def __init__(self, params):
        self.params = params
        self.memberships = params.cluster
        self.arm_count = self.params.edDict[0].actions
        self.ucb_c = 2
        self.Q = np.zeros(self.arm_count) # q-value of actions
        self.N = np.zeros(self.arm_count) + 0.0001 # action count
        self.timestep = 1

    def get_action(self):
        ln_timestep = np.log(np.full(self.arm_count, self.timestep))
        confidence = self.ucb_c * np.sqrt(ln_timestep/self.N)
        action = np.argmax(self.Q + confidence)
        self.timestep += 1
        return action

    def update(self, ed):
        self.N[ed.action] += 1 # increment action count
        self.Q[ed.action] += 1/self.N[ed.action] * (ed.dr_mean - self.Q[ed.action])
