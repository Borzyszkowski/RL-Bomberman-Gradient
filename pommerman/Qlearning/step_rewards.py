import numpy as np
from pommerman.constants import *
import operator

class StepRewards():

    # Rewards
    KILL_ENEMY = 0.3
    EXTRA_BLAST_STRENGTH = 0.12
    EXTRA_BOMB = 0.12
    EXTRA_KICK = 0.12
    PLANT_BOMB_NEAR_WOOD = 0.04
    PLANT_BOMB_NEAR_ENEMY = 0.1
    # Pentalties
    NOT_USING_AMMO = -0.01
    NOT_MOVING = -0.005
    NUM_OF_ACTIONS_WITH_NOT_USING_AMMO = 12
    NUM_OF_ACTIONS_WITH_NOT_CONSEQ_MOVE = 10

    def __init__(self, obs_prev = None, action_prev = None):
        self.reset(obs_prev, action_prev)

    def reset(self, obs_prev = None, action_prev = None):
        self.not_using_ammo = 0
        self.same_move = 0
        self.obs_prev = obs_prev
        self.action_prev = action_prev
        self.bomb_planted = False
        self.kick_picked = False
        self.bomb_capacity = 1
        self.dead_enemies = {}
        self.my_bomb_pos = None
        self.victim = 0
        self.enemy_near_flame = 0
        self.my_flames = []

    def kill_reward(self, observation, rewards):
        # Check if any of enemies has died
        self.check_enemies_on_board(observation)
        # Check if enemies are on fire in few seconds
        self.enemy_on_flames(observation)
        # Check if enemy has entered the field of destruction
        self.enemy_step_into_flames(observation)
        if str(self.victim) in self.dead_enemies:
            # print("ENEMY KILLED")
            rewards['enemy_killed'] = StepRewards.KILL_ENEMY
            self.victim = 0
        if str(self.enemy_near_flame) in self.dead_enemies and not self.victim:
            # print("ENEMY WENT INTO FLAME!")
            rewards['enemy_killed'] = StepRewards.KILL_ENEMY
            self.enemy_near_flame = 0

    def enemy_step_into_flames(self, observation):
        if self.my_flames:
            surrnd = []
            for flame in self.my_flames:
                surrnd += self.get_surroundings(flame)
            surrnd = list(set(surrnd))
            for on_pos in surrnd:
                obj = self.obs_prev['board'][on_pos]
                if obj > 10:
                    self.enemy_near_flame = obj

    def enemy_on_flames(self, observation):
        future_flames = self.get_flames(observation)
        all_flames = np.argwhere(observation['board'] == Item.Flames.value)
        if future_flames:
            for flame in future_flames:
                if self.obs_prev['board'][flame] > 10:
                    print("ENEMY ON MY FLAMES")
                    self.victim = self.obs_prev['board'][flame]
                flame_pos = list(flame)
                if flame_pos in all_flames.tolist():
                    self.my_flames.append(tuple(flame_pos))

        if all(list(flame) not in all_flames.tolist() for flame in self.my_flames):
            self.my_flames = []

    def get_flames(self, observation, timestamp=4):
        if self.my_bomb_pos:
            if self.obs_prev['bomb_life'][self.my_bomb_pos] < timestamp:
                blast_power = observation['blast_strength']
                my_flames = [(self.my_bomb_pos[0], self.my_bomb_pos[1])]
                for i in range(1, blast_power):
                    if self.my_bomb_pos[0] - i >= 0:
                        my_flames.append((self.my_bomb_pos[0] - i, self.my_bomb_pos[1]))
                    if self.my_bomb_pos[0] + i < BOARD_SIZE:
                        my_flames.append((self.my_bomb_pos[0] + i, self.my_bomb_pos[1]))
                    if self.my_bomb_pos[1] - i >= 0:
                        my_flames.append((self.my_bomb_pos[0], self.my_bomb_pos[1] - i))
                    if self.my_bomb_pos[1] + i < BOARD_SIZE:
                        my_flames.append((self.my_bomb_pos[0], self.my_bomb_pos[1] + i))

                if observation['board'][self.my_bomb_pos] == Item.Flames.value:
                    self.my_bomb_pos = None

                return my_flames

        return None

    def get_surroundings(self, position, range=4):
        surr_scratch = [(0,1), (1,0), (-1, 0), (0, -1)]
        if range == 8:
            surr_scratch += [(1, 1), (-1, 1), (-1, -1), (1, -1)]
        surroundings = []
        for el in surr_scratch:
            tuple_sums = tuple(map(operator.add, el, position))
            if 0 <= tuple_sums[0] < BOARD_SIZE and 0 <= tuple_sums[1] < BOARD_SIZE:
                surroundings.append(tuple_sums)

        return surroundings

    def bomb_planting_reward(self, rewards, observation):
        # Plant a bomb near wood or enemy
        rewards['bomb_near_wood'] = 0
        rewards['bomb_near_enemy'] = 0
        if observation['ammo'] < self.obs_prev['ammo']:
            self.not_using_ammo = 0
            self.my_bomb_pos = self.obs_prev['position']
            # self.future_flames = self.get_flames(observation)
            surrnd = self.get_surroundings(self.my_bomb_pos)
            for on_pos in surrnd:
                if self.obs_prev['board'][on_pos] == 2:
                    rewards['bomb_near_wood'] += StepRewards.PLANT_BOMB_NEAR_WOOD
                    # print("BOMB NEAR WOOD")
                if self.obs_prev['board'][on_pos] > 10:
                    rewards['bomb_near_enemy'] += StepRewards.PLANT_BOMB_NEAR_ENEMY
                    # print("BOMB NEAR ENEMY")
        elif observation['ammo'] == self.obs_prev['ammo']:
            self.not_using_ammo += 1

    def bonus_reward(self, rewards, observation):
        # Reward for picking up extra blast strength
        if observation['blast_strength'] > self.obs_prev['blast_strength']:
            rewards['extra_blast'] = StepRewards.EXTRA_BLAST_STRENGTH
            # print("GOT EXTRA BLAST")
        # Reward for picking up extra_bomb_bonus
        if observation['ammo'] > self.obs_prev['ammo'] and observation['ammo'] > self.bomb_capacity:
            rewards['extra_bomb'] = StepRewards.EXTRA_BOMB
            self.bomb_capacity += 1
            # print("EXTRA BOMB")
        # Reward for picking up can_kick_bonus
        if observation['can_kick'] == True and self.kick_picked == False:
            rewards['extra_kick'] = StepRewards.EXTRA_KICK
            self.kick_picked = True
            print("GOT EXTRA KICK")

    def penalty_conseq_move(self, rewards, observation):
        if self.not_using_ammo > StepRewards.NUM_OF_ACTIONS_WITH_NOT_USING_AMMO:
            rewards['not_using_ammo'] = StepRewards.NOT_USING_AMMO
            self.not_using_ammo = 0
            # print("PENALTY FOR NOT USING AMMO")
        if observation['position'] == self.obs_prev['position']:
            self.same_move += 1
        else:
            self.same_move = 0
        if self.same_move > StepRewards.NUM_OF_ACTIONS_WITH_NOT_CONSEQ_MOVE:
            rewards['same_move'] = StepRewards.NOT_MOVING
            self.same_move = 0
            # print("PENALTY FOR NOT MOVING")

    def check_enemies_on_board(self, observation):
        if Item.Agent1.value not in observation['board']:
            self.dead_enemies['11'] = 1
        if Item.Agent2.value not in observation['board']:
            self.dead_enemies['12'] = 1
        if Item.Agent3.value not in observation['board']:
            self.dead_enemies['13'] = 1

    def get_rewards(self, obs, action, reward):
        # If it is first step, lets move on
        if self.obs_prev is None:
            self.obs_prev = obs
            self.action_prev = action
            return reward, None

        # If our agent died or won a game return actual reward
        if reward >= 1.0 or reward <= -1.0:
            return reward, None

        # Rewards collected in one step
        new_rewards = {}
        # Grant a reward for picking up a bonus
        self.bonus_reward(new_rewards, obs)
        # Grant a reward for planting a bomb in right place
        self.bomb_planting_reward(new_rewards, obs)
        # Punish for not consequency move(same move or not planting any bomb in a sequence)
        self.penalty_conseq_move(new_rewards, obs)
        # Reward for killing an enemy:
        self.kill_reward(obs, new_rewards)

        # Update observation
        self.obs_prev = obs
        self.action_prev = action

        # Sum all collected rewards
        rewards = reward + sum(new_rewards.values())
        # Limit output reward
        rewards = np.clip(rewards, -0.9, 0.9)

        return rewards, new_rewards
