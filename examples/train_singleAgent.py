"""Double dueling DQN Agent"""

import pommerman
from pommerman import agents

import torch
import numpy as np
import random
from collections import namedtuple
import getopt
import sys

from tensorboardX import SummaryWriter  # requires protobuf newest than 3.5


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def save_checkpoint(state, agent):
    filename = "Checkpoints/" + agent + '_game #' + str(state['epoch']) + ".pth"
    torch.save(state, filename)  


def load_checkpoint(agent, path):
    checkpoint = torch.load(path)
    agent.Q.load_state_dict(checkpoint['state_dict_Q'])
    agent.target_Q.load_state_dict(checkpoint['state_dict_target_Q'])
    agent.optimizer.load_state_dict(checkpoint['optimizer'])
    
    return checkpoint['epoch']


def main(argv):
    checkpoint_file_path = ''
    always_render = False
    force_restart_on_death = False
    try:
        opts, args = getopt.getopt(argv, "hrRc:a:", ["checkpoint=", "agent=", "restart_on_death"])
    except getopt.GetoptError:
        print('Error in command arguments. Run this for help:\n\ttrain_singleAgent.py -h')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print("train_singleAgent.py" +
                  "\n-c <checkpointfile> => Resume training from a saved checkpoint" +
                  "\n-a(--agent) <agent version> => Version of agent to train (default=0)" +
                  "   \n-r => always_render" +
                  "\n-R(--restart_on_death) => always_render")
            sys.exit()
        elif opt in ("-c", "--checkpoint"):
            checkpoint_file_path = arg
        elif opt == '-r':
            always_render = True
        elif opt in ("-R", "--restart_on_death"):
            force_restart_on_death = True

    # Create a set of 4 agents
    agent_list = [
        agents.SimpleAgent(),
        agents.SimpleAgent(),
        agents.SimpleAgent(),
        agents.SPN_agent()]

    # Make the "Team" environment using the agent list
    env = pommerman.make('PommeFFACompetitionFast-v0', agent_list)
    memory = ReplayMemory(100000)
    batch_size = 128
    epsilon = 1
    start_epoch = 0
    end_epoch = 5750  # add number of games here

    # Writer to log data to tensorboard
    writer = SummaryWriter('runs')

    if checkpoint_file_path != '':
        start_epoch = load_checkpoint(agent_list[3], checkpoint_file_path)

    # Run the episodes just like OpenAI Gym
    for i in range(start_epoch, end_epoch):
        state = env.reset()
        steps_per_episode = 0
        done = False
        total_reward = [0] * len(agent_list)
        action_histo = np.zeros(6)
        epsilon *= 0.995
        alive_steps = 0
        while not done and (not force_restart_on_death or agent_list[3]._character.is_alive):
            if always_render:
                env.render()
            # Set epsilon for our learning agent
            agent_list[3].epsilon = max(epsilon, 0.1)
            
            actions = env.act(state)
            state, reward, done, info = env.step(actions)

            # Fill replay memory for our learning agent
            memory.push(agent_list[3].Input, torch.LongTensor([actions[3]]),
                        torch.from_numpy(agent_list[3].prep_input(state[3])).type(torch.FloatTensor),
                        torch.Tensor([reward[3]]),
                        torch.Tensor([done]))

            # Save info about our leaning agent
            action_histo[actions[3]] += 1
            alive_steps += 1
            total_reward = [x + y for x, y in zip(total_reward, reward)]
            steps_per_episode += 1

        # Log infos about our leaning agent to tensorboad
        writer.add_scalars('data/actions', {'stop': action_histo[0], 'up': action_histo[1], 'down': action_histo[2],
                                            'left': action_histo[3], 'right': action_histo[4],
                                            'bomb': action_histo[5]}, i)
        writer.add_scalar('data/alive_steps', alive_steps, i)
        writer.add_scalar('data/epsilon', agent_list[3].epsilon, i)
        writer.add_scalar('data/memory', memory.__len__(), i)

        # Creates a dictionary with agent name and rewards to be logged on tensorboard
        total_reward_list = []
        for j in range(len(total_reward)):
            total_reward_list.append((type(agent_list[j]).__name__ + '(' + str(j) + ')', total_reward[j]))
        writer.add_scalars('data/rewards', dict(total_reward_list), i)

        spinInput = agent_list[3].Input
        writer.add_image('end_img', spinInput.reshape(spinInput.shape[1], spinInput.shape[2], spinInput.shape[3]), i)

        if 'winners' in info:
            print("Episode : {}, steps_per_episode: {}, winner: {}, reward: {}, total_reward: {}"
                  .format(i, steps_per_episode, info['winners'], reward, total_reward))
        else:
            print("Episode : {}, steps_per_episode: {}, result: draw, reward: {}, total_reward: {}"
                  .format(i, steps_per_episode, reward, total_reward))

        if memory.__len__() > 10000:
            batch = memory.sample(batch_size)
            agent_list[3].backward(batch)
        if i > 0 and i % 100 == 0:
            save_checkpoint({
                    'epoch': i + 1,
                    'arch': 0,
                    'state_dict_Q': agent_list[3].Q.state_dict(),
                    'state_dict_target_Q': agent_list[3].target_Q.state_dict(),
                    'best_prec1': 0,
                    'optimizer': agent_list[3].optimizer.state_dict(),
                }, agent_list[3].__class__.__name__)
    env.close()

    save_checkpoint({
            'epoch': end_epoch + 1,
            'arch': 0,
            'state_dict_Q': agent_list[3].Q.state_dict(),
            'state_dict_target_Q': agent_list[3].target_Q.state_dict(),
            'best_prec1': 0,
            'optimizer': agent_list[3].optimizer.state_dict(),
        }, agent_list[3].__class__.__name__)

    writer.close()


if __name__ == '__main__':
    main(sys.argv[1:])
