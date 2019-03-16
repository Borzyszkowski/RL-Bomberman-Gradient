from .env_wrapper import EnvWrapper
from .rewards import Rewards


class EnvWrapperRS(EnvWrapper):
    def __init__(self, gym, board_size):
        super(EnvWrapperRS, self).__init__(gym, board_size)
        self.rewardShaping = Rewards()

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

        action = all_actions[self.gym.training_agent]
        # agent_state = self.featurize(state[self.gym.training_agent])
        agent_reward = reward[self.gym.training_agent]
        # custom featurize with changes on board
        agent_state = self.custom_featurize(state[self.gym.training_agent])

        agent_reward, reward_info = self.rewardShaping.get_rewards(obs[self.gym.training_agent],
                                                                action, agent_reward)

        return agent_state, agent_reward, terminal, info

    def reset(self):
        """
        Resets the state of the environment and returns an initial observation.
        # Returns
            observation (object): The initial observation of the space. Initial reward is assumed to be 0.
        """
        obs = self.gym.reset()
        self.rewardShaping.reset()
        agent_obs = self.featurize(obs[self.gym.training_agent])
        return agent_obs
