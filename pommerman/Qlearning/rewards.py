import numpy as np
from pommerman.constants import *

class Rewards:

    def __init__(self, obs_prev=None, action_prev=None):
        self.reset(obs_prev, action_prev)

    def reset(self, obs_prev=None, action_prev=None):
        self.obs_prev = obs_prev
        self.action_prev = action_prev
        self.not_using_ammo = 0
        self.making_the_same_move = 0
        self.dist_to_bomb_prev = 0
        self.closestEnemyIdPrev = -1
        self.closestEnemyDistPrev = float("inf")

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
        PICK_UP_BONUS = 0.1
        MOBILITY_REWARD = 0.01
        DIST_TO_BOMB_INCREASED = 0.05
        # Penalties
        MAKING_THE_SAME_MOVE = -0.0001
        NOT_USING_AMMO = -0.0005
        ON_FLAMES = -0.0001
        CATCH_ENEMY = 0.001

        position_prev = np.array(self.obs_prev['position'])
        position_now = np.array(obs_now['position'])
        euclidean_dist = np.linalg.norm(position_now - position_prev)
        # Reward for making a move
        if euclidean_dist != 0:
            new_rewards['mobility'] = MOBILITY_REWARD
        else:
            new_rewards['mobility'] = 0

        # Reward for picking a bonus - not working :(
        current_pos = obs_now['position']
        item_on_next_position = self.obs_prev['board'][current_pos]
        # 6 - extra bomb, 7 - extra range, 8 - extra kick
        if 6 <= item_on_next_position <= 8:
            new_rewards['bonus'] = PICK_UP_BONUS
            print("picked UP bonus")
        else:
            new_rewards['bonus'] = 0


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

        # Rewards for planting a bomb
        # plant a bomb: + based on value of the bombing position
        bombs_pose = np.argwhere(obs_now['bomb_life'] != 0)
        if obs_now['ammo'] < self.obs_prev['ammo']:
            surroundings = [(-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0)]
            mybomb_pose = self.obs_prev['position']  # equal to agent previous position
            # validate if the bomb actually exists there
            found_the_bomb = False
            for bp in bombs_pose:
                if np.equal(bp, mybomb_pose).all():
                    found_the_bomb = True
                    break
            assert found_the_bomb  # end of validation
            nr_woods = 0
            nr_enemies = 0
            for p in surroundings:
                cell_pose = (mybomb_pose[0] + p[0], mybomb_pose[1] + p[1])
                if cell_pose[0] > 10 or cell_pose[1] > 10:  # bigger than board size
                    continue
                # print(obs_now['board'][cell_pose])
                nr_woods += obs_now['board'][cell_pose] == Item.Wood.value
                nr_enemies += obs_now['board'][cell_pose] in [e.value for e in obs_now['enemies']]
            # print("nr woods: ", nr_woods)
            # print("nr enemies: ", nr_enemies)
            assert nr_woods + nr_enemies < 10
            new_rewards['plantbomb'] = PLANT_A_BOMB_NEAR_WOOD * nr_woods + PLANT_A_BOMB_NEAR_ENEMY * nr_enemies

        # on Flames: - if agent on any blast direction
        for bp in bombs_pose:
            def rot_deg90cw(point):
                new_point = [0, 0]
                new_point[0] = point[1]
                new_point[1] = -point[0]
                return new_point

            # print(type(bp))
            factor = 1 / obs_now['bomb_life'][tuple(bp)]  # inverse of time left
            blast_strength = obs_now['bomb_blast_strength'][tuple(bp)]

            # blast directions
            blast_N = Point(0, 1).scale(blast_strength)
            blast_S = Point(0, -1).scale(blast_strength)
            blast_W = Point(-1, 0).scale(blast_strength)
            blast_E = Point(1, 0).scale(blast_strength)

            # agent on blast direction?
            bpPose = rot_deg90cw(bp)
            myPose = rot_deg90cw(obs_now['position'])
            myPose = Point(myPose[0] - bpPose[0], myPose[1] - bpPose[1])  # my pose relative to the bomb!
            onBlastDirect = is_between(blast_N, blast_S, myPose) or \
                            is_between(blast_W, blast_E, myPose)
            if onBlastDirect:
                # print("time: ", obs_now['bomb_life'][tuple(bp)])
                # print("on blast: ", factor)
                new_rewards['onflame'] = ON_FLAMES * factor

            # catch enemy: + if closing distance with the nearest enemy

        def closestEnemy():
            myPose = obs_now['position']
            closestEnemyId = -1
            closestEnemyDist = float("inf")
            for e in obs_now['enemies']:
                enemyPose = np.argwhere(obs_now['board'] == e.value)
                if len(enemyPose) == 0:
                    continue
                dist2Enemy = np.linalg.norm(myPose - enemyPose)
                if dist2Enemy <= closestEnemyDist:
                    closestEnemyId = e.value
                    closestEnemyDist = dist2Enemy
            return closestEnemyId, closestEnemyDist

        closestEnemyId_cur, closestEnemyDist_cur = closestEnemy()
        if self.closestEnemyIdPrev != closestEnemyId_cur:
            self.closestEnemyIdPrev = closestEnemyId_cur
            self.closestEnemyDistPrev = closestEnemyDist_cur
        else:
            CATCHING_TRHE = 4  # not too close
            if closestEnemyDist_cur < self.closestEnemyDistPrev and \
                    closestEnemyDist_cur < CATCHING_TRHE:
                new_rewards['catchenemy'] = CATCH_ENEMY
                self.closestEnemyDistPrev = closestEnemyDist_cur
            if closestEnemyDist_cur <= 1.1:  # got that close
                self.closestEnemyDistPrev = float("inf")

        # Update observations
        self.obs_prev = obs_now
        self.action_prev = action_now

        # Sum all rewards
        rewards = reward + sum(new_rewards.values())
        # Limit output reward
        rewards = np.clip(rewards, -0.8, 0.8)

        return rewards, new_rewards


def is_between(a, b, c):
    crossproduct = (c.y - a.y) * (b.x - a.x) - (c.x - a.x) * (b.y - a.y)
    epsilon = 0.0001
    # compare versus epsilon for floating point values, or != 0 if using integers
    if abs(crossproduct) > epsilon:
        return False

    dotproduct = (c.x - a.x) * (b.x - a.x) + (c.y - a.y) * (b.y - a.y)
    if dotproduct < 0:
        return False

    squaredlengthba = (b.x - a.x) * (b.x - a.x) + (b.y - a.y) * (b.y - a.y)
    if dotproduct > squaredlengthba:
        return False

    return True


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def scale(self, s):
        self.x *= s
        self.y *= s
        return self