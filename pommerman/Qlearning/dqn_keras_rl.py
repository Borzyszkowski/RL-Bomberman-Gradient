import numpy as np
from pommerman.agents import BaseAgent, SimpleAgent
from pommerman.Qlearning.boardLogger import TensorboardLogger

from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, Dropout
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy, MaxBoltzmannQPolicy, LinearAnnealedPolicy
from rl.memory import SequentialMemory
from rl.callbacks import FileLogger, ModelIntervalCheckpoint

from pommerman.envs.v0 import Pomme
from pommerman.configs import ffa_v0_fast_env
from pommerman.Qlearning.env_wrapper import CustomProcessor, EnvWrapper
from pommerman.Qlearning.env_with_rewards import EnvWrapperRS
from pommerman.constants import *

import tensorflow as tf


history_length = 4
# BOARD_SIZE = 11
view_size = BOARD_SIZE*2 - 1
n_channels = 18
config = ffa_v0_fast_env()
env = Pomme(**config["env_kwargs"])
ACTIONS = ['stop', 'up', 'down', 'left', 'right', 'bomb']
NUM_OF_ACTIONS = len(ACTIONS)
NUMBER_OF_STEPS = 5e6


# Just formal to create enviroment
class DQN(BaseAgent):
    def act(self, obs, action_space):
        pass


def create_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=2, strides=(1,1), input_shape=(view_size, view_size, n_channels*history_length),
                     activation='relu'))
    model.add(Conv2D(64, kernel_size=2, strides=(1,1), activation='relu'))
    model.add(Conv2D(64, kernel_size=2, strides=(1,1), activation='relu'))
    model.add(Flatten())
    model.add(Dense(units=128, activation='relu'))#original
    model.add(Dense(units=128, activation='relu'))#was one Dense(units=512, activation='relu') if u want to run my weights
    model.add(Dense(units=NUM_OF_ACTIONS, activation='linear'))
    model.compile(optimizer=Adam(lr=0.0005), loss=tf.losses.huber_loss, metrics=['mae'])
    return model


def load_trained_model(weights_path = './models/dqn_agent_fullmodel.hdf5'):
    model = create_model()
    model.load_weights(filepath=weights_path)
    return model


def set_pommerman_env(agent_id=0):
    # Instantiate the environment
    config = ffa_v0_fast_env()
    env = Pomme(**config["env_kwargs"])

    np.random.seed(0)
    env.seed(0)
    # Add 3 Simple Agents and 1 DQN agent
    agents = [DQN(config["agent"](agent_id, config["game_type"])) if i == agent_id else SimpleAgent(
        config["agent"](i, config["game_type"])) for i in range(4)]
    env.set_agents(agents)
    env.set_training_agent(agents[agent_id].agent_id)  # training_agent is only dqn agent
    env.set_init_game_state(None)

    return env


def create_dqn(model,
            log_interval=50000,
            model_name='dqn_agent_checkpoint',
            file_log_path='./logs/log.txt',
            tensorboard_path='./logs/tensorboard/'):
    model_path = './models/' + model_name + '.h5'
    file_logger = FileLogger(file_log_path, interval=log_interval)
    checkpoint = ModelIntervalCheckpoint(model_path, interval=log_interval)
    tensorboard = TensorboardLogger(tensorboard_path)
    callbacks = [file_logger, checkpoint, tensorboard]

    # Use 4 last observation - history_length = 4
    memory = SequentialMemory(limit=500000, window_length=history_length)

    # Use combine of BoltzmannQPolicy and EpsGreedyQPolicy
    policy = MaxBoltzmannQPolicy()

    # Set epsilon to 1.0 and decrease it over every step to stop taking random action when map is explored
    policy = LinearAnnealedPolicy(inner_policy=policy,
                                  attr='eps',
                                  value_max=1.0,
                                  value_min=0.1,
                                  value_test=0.04,
                                  nb_steps=NUMBER_OF_STEPS)

    # Create an instance of DQNAgent from keras-rl
    dqn = DQNAgent(model=model,
                   nb_actions=env.action_space.n,
                   memory=memory,
                   policy=policy,
                   processor=CustomProcessor(),
                   nb_steps_warmup=512,
                   enable_dueling_network=True,
                   dueling_type='avg',
                   target_model_update=5e2,
                   batch_size=32)

    dqn.compile(Adam(lr=5e-4), metrics=['mae'])

    return dqn, callbacks


if __name__ == '__main__':

    env_wrapper = EnvWrapperRS(set_pommerman_env(agent_id=0), BOARD_SIZE)
    dqn, callbacks = create_dqn(create_model())

    dqn.load_weights('./models/15_03_18-40_marcin_weight_with_rewards.h5')

    dqn.fit(env_wrapper, nb_steps=NUMBER_OF_STEPS, visualize=False, verbose=2,
                      nb_max_episode_steps=env._max_steps, callbacks=callbacks)
    dqn.model.save('./models/14_03_19.hdf5')
