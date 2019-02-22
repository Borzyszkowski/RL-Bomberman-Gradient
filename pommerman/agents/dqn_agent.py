from builtins import super

from . import BaseAgent
import tensorflow as tf
import numpy as np
from collections import deque
import random

class DQN_agent(BaseAgent):
    
    def __init__(self, model):
        super(DQN_agent, self).__init__()
        self.model = model
        self.memory = deque(maxlen=100000)
        # set hyperparameters
        self.gamma = 0.95
        self.epsilon = 1.0
        self.ep_decay = 0.999
        self.batch_size = 32

    # TODO: modify state format (for now it's only board, without any other features)
    def act(self, obs, action_space):
        if random.random() < self.epsilon:
            a = action_space.sample()
        else:
            a = np.argmax(self.model.predict(np.array([obs['board']])))
        return a
        
    # TODO: modify state format (for now it's only board, without any other features)
    def remember(self, old_state, action, reward, done, new_state):
        self.memory.append((old_state['board'], action, reward, done, new_state['board']))
        if done:
            self.epsilon *= self.ep_decay
    
    # learning from experience
    def experience_replay(self):
        if len(self.memory) > self.batch_size:
            batch = random.sample(self.memory, self.batch_size)
            for old_state, action, reward, done, new_state in batch:
                q_update = reward
                if not done:
                    q_update += self.gamma * np.max(self.model.predict(np.array([new_state])))
                q_val = self.model.predict(np.array([old_state]))[0]
                q_val[action] = q_update
                self.model.fit(np.array([old_state]), np.array([q_val]), verbose=0)

    def set_model(self, model):
        self.model = model

    def get_model(self):
        return self.model

    def save_weights(self, file_name):
        self.model.save_weights(file_name)
        
    def restore_weights(self, file_name):
        self.model.load_weights(file_name)