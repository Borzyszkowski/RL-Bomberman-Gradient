import pommerman
from pommerman import agents
import tensorflow as tf

def main():
    # Create agent able to learn
    model = create_model()
    dqn_agent = agents.DQN_agent(model)
    dqn_agent_index = 1

    # Create a set of two agents
    agent_list = [
        agents.SimpleAgent(),
        dqn_agent
    ]

    # Make the "Free-For-All" environment using the agent list
    env = pommerman.make('PommeFFACompetition-v0', agent_list)

    # Run the episodes just like OpenAI Gym
    for episode in range(5):
        state = env.reset()
        done = False
        while not done:
            # env.render()
            actions = env.act(state)
            new_state, reward, done, info = env.step(actions)

            # agent learning
            dqn_agent.remember(
                state[dqn_agent_index], actions[dqn_agent_index],
                reward[dqn_agent_index], done, new_state[dqn_agent_index]
            )
            dqn_agent.experience_replay()

            state = new_state

        print('Episode {} finished'.format(episode))
    env.close()


def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(6, activation='softmax')
    ])
    model.compile(optimizer=tf.train.AdamOptimizer(0.001), loss='mse', metrics=['mae'])
    return model

if __name__ == '__main__':
    main()
