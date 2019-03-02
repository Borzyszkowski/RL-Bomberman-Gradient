from builtins import super

from . import BaseAgent
import tensorflow as tf
import numpy as np
from collections import deque
import random

class DQN_agent(BaseAgent):
    
    def __init__(self, model, gamma=0.95, memory_length=10000, 
        epsilon=1.0, ep_decay=0.999, batch_size=32):
        super(DQN_agent, self).__init__()
        self.model = model
        self.memory = deque(maxlen=memory_length)
        # set hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon
        self.ep_decay = ep_decay
        self.batch_size = batch_size

    # TODO: modify state format (for now it's only board, without any other features)
    def act(self, obs, action_space):
        if random.random() < self.epsilon:
            a = action_space.sample()
        else:
            a = np.argmax(self.model.predict(np.array([self.preprocess_observation(obs)])))
        return a
        
    def remember(self, obs, action, reward, done, new_obs):
        self.memory.append((self.preprocess_observation(obs), action, reward, done, self.preprocess_observation(new_obs)))
        if done:
            self.epsilon *= self.ep_decay
    
    # learning from experience
    def experience_replay(self):
        if len(self.memory) > self.batch_size:
            batch = random.sample(self.memory, self.batch_size)
            q_vals = []
            old_states = []
            for old_state, action, reward, done, new_state in batch:
                old_states.append(old_state)
                q_update = reward
                if not done:
                    q_update += self.gamma * np.max(self.model.predict(np.array([new_state])))
                q_val = self.model.predict(np.array([old_state]))[0]
                q_val[action] = q_update
                q_vals.append(q_val)
            old_states = np.array(old_states)
            q_vals = np.array(q_vals)
            self.model.fit(old_states, q_vals, verbose=0)

    # prepare state as stack of 11x11 matrices
    def preprocess_observation(self, obs):
        # list for matrices
        prepared_features_list = []

        # board matrixs with enemies as 11 and agent as 10
        board_features = obs['board']
        agent_x, agent_y = obs['position']
        for i in range(len(board_features)):
            for j in range(len(board_features[i])):
                if 10 <= board_features[i][j] <= 13:
                    board_features[i][j] = 11
                if i == agent_y and j == agent_x:
                    board_features[i][j] = 10
        board_features = tf.constant(board_features, dtype=tf.float32)
        board_features = tf.reshape(board_features, (11, 11, 1))
        prepared_features_list.append(board_features)
    
        # bomb_blast_strength and bomb_life matrices
        sparse_features = ['bomb_blast_strength', 'bomb_life']
        for feature in sparse_features:
            prepared_feature = tf.constant(obs[feature], dtype=tf.float32)
            prepared_feature = tf.reshape(prepared_feature, (11, 11, 1))
            prepared_features_list.append(prepared_feature)
        # blast strength as a matrix
        blast_strength_feature = tf.ones((11, 11, 1), dtype=tf.float32) * obs['blast_strength']
        prepared_features_list.append(blast_strength_feature)
        # can kick as a matrix
        can_kick = 1 if obs['can_kick'] else 0.0
        can_kick_feature = tf.ones((11, 11, 1), dtype=tf.float32) * can_kick
        prepared_features_list.append(can_kick_feature)
        # ammo number as a matrix
        ammo_feature = tf.ones((11, 11, 1), dtype=tf.float32) * obs['ammo']
        prepared_features_list.append(ammo_feature)
        # run session on prepared_features_list to have an actual values  
        with tf.Session() as sess:
            state = sess.run(tf.concat(prepared_features_list, axis=2))
            sess.close()
        return state

    def set_model(self, model):
        self.model = model

    def get_model(self):
        return self.model

    def save_weights(self, file_name):
        self.model.save_weights(file_name)
        
    def restore_weights(self, file_name):
        self.model.load_weights(file_name)

