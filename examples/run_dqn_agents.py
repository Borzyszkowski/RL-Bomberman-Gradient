'''An example to show how to set up an pommerman game programmatically'''
import pommerman
from pommerman import agents
from learning_run import create_model


def main():
    '''Simple function to bootstrap a game.

       Use this as an example to set up your training env.
    '''
    # Print all possible environments in the Pommerman registry
    print(pommerman.REGISTRY)

    # create model
    loaded_model = create_model()
    # load weights into new model
    loaded_model.load_weights("./weights/model_2_episode_58")
    dqn_agent1 = agents.DQN_agent(loaded_model)
    dqn_agent2 = agents.DQN_agent(loaded_model)
    dqn_agent3 = agents.DQN_agent(loaded_model)
    dqn_agent4 = agents.DQN_agent(loaded_model)

    # Create a set of agents (exactly four)
    agent_list = [
        dqn_agent1,  # Agent 0 - left top
        dqn_agent2,  # Agent 1 - left bottom
        dqn_agent3,  # Agent 2 - right bottom
        dqn_agent4  # Agent 3 - right top
    ]
    # Make the "Free-For-All" environment using the agent list
    env = pommerman.make('PommeFFACompetition-v0', agent_list)

    # Run the episodes just like OpenAI Gym
    for i_episode in range(1):
        state = env.reset()
        done = False
        while not done:
            env.render()
            actions = env.act(state)
            state, reward, done, info = env.step(actions)
        print('Episode {} finished, winner: {}'.format(i_episode, info['winners']))
    env.close()


if __name__ == '__main__':
    main()
