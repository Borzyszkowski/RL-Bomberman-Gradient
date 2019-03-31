from rl.core import Env, Processor
import numpy as np
from pommerman import utility
import os, sys
sys.path.append('./evaluation_utils.py/')
from .evaluation_utils import *


class EnvWrapper(Env):
    """The abstract environment class that is used by all agents. This class has the exact
        same API that OpenAI Gym uses so that integrating with it is trivial. In contrast to the
        OpenAI Gym implementation, this class only defines the abstract methods without any actual
        implementation.
        To implement your own environment, you need to define the following methods:
        - `step`
        - `reset`
        - `render`
        - `close`
        Refer to the [Gym documentation](https://gym.openai.com/docs/#environments).
        """
    reward_range = (-1, 1)
    action_space = None
    observation_space = None



    def __init__(self, gym, board_size):
        self.gym = gym
        self.action_space = gym.action_space
        self.observation_space = gym.observation_space
        self.reward_range = gym.reward_range
        self.board_size = board_size

    def step(self, action):
        """Run one timestep of the environment's dynamics.
        Accepts an action and returns a tuple (observation, reward, done, info).
        # Arguments
            action (object): An action provided by the environment.
        # Returns
            observation (object): Agent's observation of the current environment.
            reward (float) : Amount of reward returned after previous action.
            done (boolean): Whether the episode has ended, in which case further step() calls will return undefined results.
            info (dict): Contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).
        """
        obs = self.gym.get_observations()
        all_actions = self.gym.act(obs)
        all_actions.insert(self.gym.training_agent, action)
        state, reward, terminal, info = self.gym.step(all_actions)
        agent_state = self.custom_featurize(state[self.gym.training_agent])

        #         agent_state_history = self.make_observation(agent_state, self.step)
        agent_reward = reward[self.gym.training_agent]

        #         self.step += 1
        return agent_state, agent_reward, terminal, info

    def reset(self):
        """
        Resets the state of the environment and returns an initial observation.
        # Returns
            observation (object): The initial observation of the space. Initial reward is assumed to be 0.
        """
        obs = self.gym.reset()
        #         self.step = 1
        agent_obs = self.custom_featurize(obs[self.gym.training_agent])
        #         self.observations_history = [agent_obs]
        return agent_obs

    def render(self, mode='human', close=False):
        """Renders the environment.
        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.)
        # Arguments
            mode (str): The mode to render with.
            close (bool): Close all open renderings.
        """
        self.gym.render(mode=mode, close=close)

    def close(self):
        """Override in your subclass to perform any necessary cleanup.
        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        self.gym.close()

    def seed(self, seed=None):
        """Sets the seed for this env's random number generator(s).
        # Returns
            Returns the list of seeds used in this env's random number generators
        """
        raise self.gym.seed(seed)

    def configure(self, *args, **kwargs):
        """Provides runtime configuration to the environment.
        This configuration should consist of data that tells your
        environment how to run (such as an address of a remote server,
        or path to your ImageNet data). It should not affect the
        semantics of the environment.
        """
        raise NotImplementedError()

    def featurize(self, obs):
        return featurize(obs, center=True, crop=False)

    def custom_featurize(self, obs):
        board_features = obs['board'].copy()
        # get board coordinates of our agent
        agent_y, agent_x = obs['position']
        for i in range(len(board_features)):
            for j in range(len(board_features[i])):
                # set enemies on board to 11
                if 10 <= board_features[i][j] <= 13:
                    board_features[i][j] = 11
                # set our agent to 10
                if i == agent_y and j == agent_x:
                    board_features[i][j] = 10
        observation = obs.copy()
        observation['board'] = board_features
        return featurize(observation, center=True, crop=False)



    def __del__(self):
        self.close()

    def __str__(self):
        return '<{} instance>'.format(type(self).__name__)


class CustomProcessor(Processor):
    def process_state_batch(self, batch):
        """Processes an entire batch of states and returns it.
        # Arguments
            batch (list): List of states
        # Returns
            Processed list of states
        """
        #         batch = np.squeeze(batch, axis=1)
        batch = np.array([np.concatenate(obs, axis=-1) for obs in batch])
        return batch

    def process_info(self, info):
        """Processes the info as obtained from the environment for use in an agent and
        returns it.
        """
        info['result'] = info['result'].value
        return info