import numpy as np


class Rewards:

    def __init__(self, obs_prev=None, action_prev=None):
        self.obs_prev = obs_prev
        self.action_prev = action_prev
        self.not_using_ammo = 0
        self.making_the_same_move = 0
        self.dist_to_bomb_prev = 0

    def reset(self, obs_prev=None, action_prev=None):
        self.obs_prev = obs_prev
        self.action_prev = action_prev
        self.not_using_ammo = 0
        self.making_the_same_move = 0
        self.dist_to_bomb_prev = 0

    def get_rewards(self, obs_now, action_now, reward):
        if self.obs_prev is None:
            self.obs_prev = obs_now
            self.action_prev = action_now
            return reward, None
        if reward == 1.0 or reward == -1.0:
            return reward, None

        new_rewards = {}
        # Rewards
        PLANT_A_BOMB_NEAR_WOOD = 0.05
        PLANT_A_BOMB_NEAR_ENEMY = 0.1
        KILL_ENEMY = 0.3
        PICK_UP_BONUS = 0.1
        MOBILITY_REWARD = 0.05
        DIST_TO_BOMB_INCREASED = 0.05
        # Penalties
        MAKING_THE_SAME_MOVE = -0.0001
        NOT_USING_AMMO = -0.0005

        position_prev = np.array(self.obs_prev['position'])
        position_now = np.array(obs_now['position'])
        euclidean_dist = np.linalg.norm(position_now - position_prev)
        # Reward for making a move
        if euclidean_dist != 0:
            new_rewards['mobility'] = MOBILITY_REWARD
        else:
            new_rewards['mobility'] = 0

        # Reward for picking a bonus
        current_pos = obs_now['position']
        item_on_next_position = self.obs_prev['board'][current_pos]
        # 6 - extra bomb, 7 - extra range, 8 - extra kick
        if 6 <= item_on_next_position <= 8:
            new_rewards['bonus'] = PICK_UP_BONUS
        else:
            new_rewards['bonus'] = 0

        # Reward for extra distance to bomb TODO
        bomb_positions = np.argwhere(obs_now['bomb_life'] != 0)
        for bomb_pos in bomb_positions:
            curr_dist = np.linalg.norm(current_pos - bomb_pos)

        # Penalty for not using ammo
        if obs_now['ammo'] == self.obs_prev['ammo']:
            self.not_using_ammo += 1
        else:
            self.not_using_ammo = 0
        if self.not_using_ammo > 13:
            new_rewards['ammousage'] = NOT_USING_AMMO

        # Penalty for making the same move
        if obs_now['position'] == self.obs_prev['position']:
            self.making_the_same_move += 1
        else:
            self.making_the_same_move = 0
        if self.making_the_same_move > 8:
            new_rewards['consequency'] = MAKING_THE_SAME_MOVE

        # Update observations
        self.obs_prev = obs_now
        self.action_prev = action_now

        # Sum all rewards
        rewards = reward + sum(new_rewards.values())
        # Limit output reward
        rewards = np.clip(rewards, -0.9, 0.9)

        return rewards, new_rewards
