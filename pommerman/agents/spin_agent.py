from pommerman import agents
import numpy as np
import torch
from torch import nn
import torch.nn.functional as torchfun
from torch.autograd import Variable
import random
from collections import namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))


class DQN(nn.Module):
    def __init__(self, dueling=True):
        super().__init__()
        self.dueling = dueling
        self.conv1 = nn.Conv2d(3, 1, 2, padding=1)
        self.fc1 = nn.Linear(144, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 6)
        if dueling:
            self.v = nn.Linear(512, 1)
    
    def forward(self, x):
        x = torchfun.relu(self.conv1(x))
        x = x.reshape(x.shape[0], x.shape[2]*x.shape[3])
        x = torchfun.relu(self.fc1(x))
        x = torchfun.relu(self.fc2(x))
        if self.dueling:
            v = self.v(x)
            a = self.fc3(x)
            q = v + a
        else:
            q = self.fc3(x)
        return q


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


class SPN_agent (agents.BaseAgent):

    def __init__(self, *args, **kwargs):
        super(SPN_agent, self).__init__(*args, **kwargs)
        self.target_Q = DQN()
        self.Q = DQN()
        self.gamma = 0.8
        self.batch_size = 128
        self.epsilon = 0.1
        hard_update(self.target_Q, self.Q)
        self.optimizer = torch.optim.Adam(self.Q.parameters(), lr=0.0001)
    
    def prep_input(self, obs):
        # Add board to input
        board = np.array(list(obs['board'].copy()))
        network_input = np.reshape(board, (1, board.shape[0], board.shape[1]))
        # Add bomb strength map
        bomb_blast_map = np.array(list(obs['bomb_blast_strength'].copy()))
        network_input = np.append(network_input, bomb_blast_map.reshape(1, bomb_blast_map.shape[0],
                                                                        bomb_blast_map.shape[1]), axis=0)
        # Add position as a board map with '1' to denote player position
        position_board = obs['board'].copy()
        for i in range(len(obs['board'])):
            for j in range(len(obs['board'])):
                if (i, j) == obs['position']:
                    position_board[i][j] = 1
                else:
                    position_board[i][j] = 0
        position_board = np.array(list(position_board))
        network_input = np.append(network_input, position_board.reshape(1, position_board.shape[0],
                                                                        position_board.shape[1]), axis=0)
        # Preparate input for convolution
        network_input = network_input.reshape(1, network_input.shape[0], network_input.shape[1],
                                              network_input.shape[2])
        
        return network_input

    def act(self, obs, action_space):#self, x, epsilon=0.1):
        self.Input = Variable(torch.from_numpy(self.prep_input(obs)).type(torch.FloatTensor))
        x = self.Input
        p = random.uniform(0, 1)

        # Return random action with probability epsilon
        if p < self.epsilon:
            action = int(np.round(random.uniform(-0.5, 5.5)))
            action = max(0, min(action, 5))
            return action

        q_sa = self.Q(x.data)
        argmax = q_sa.data.max(1)[1]
        return argmax.data.numpy()[0]
    
    def backward(self, transitions):
        batch = Transition(*zip(*transitions))

        state = Variable(torch.cat(batch.state))
        action = Variable(torch.from_numpy(np.array(batch.action)))
        next_state = Variable(torch.cat(batch.next_state))
        reward = Variable(torch.cat(batch.reward))
        done = Variable(torch.from_numpy(np.array(batch.done)))

        q_sa = self.Q(next_state).detach()
        target = self.target_Q(next_state).detach()

        _, argmax = q_sa.max(dim=1, keepdim=True)
        target = target.gather(1, argmax)

        current_qvalues = self.Q(state).gather(1, action.unsqueeze(1)).squeeze()
        y = (reward.unsqueeze(1) + self.gamma * (target * (1-done.unsqueeze(1)))).squeeze()

        loss = torchfun.smooth_l1_loss(current_qvalues, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        soft_update(self.target_Q, self.Q, 0.995)
