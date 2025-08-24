import numpy as np


class Agent:
    def __init__(self, epsilon=0.1, arms=10):
        self.epsilon = epsilon
        self.arms = arms
        self.Qs = np.zeros(arms)
        self.Ns = np.zeros(arms)

    def update(self, action, reward):
        self.Ns[action] += 1
        self.Qs[action] += (reward - self.Qs[action]) / self.Ns[action]

    def get_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.arms)
        return np.argmax(self.Qs)
