import pommerman
from pommerman import agents
import tensorflow as tf

def main():
    # Create agent able to learn
    # model = create_model()
    # dqn_agent = agents.DQN_agent(model, epsilon=0.5)
    # dqn_agent_index = 0

    # create four models to check which is winning most times
    models = create_four_test_models()
    dqn_agents = []
    for i in range(4):
        dqn_agents.append(agents.DQN_agent(models[i], epsilon=0.5))

    # Create a set of two agents
    agent_list = [
        dqn_agents[0],
        dqn_agents[1],
        dqn_agents[2],
        dqn_agents[3]
    ]

    # Make the "Free-For-All" environment using the agent list
    env = pommerman.make('PommeFFACompetitionFast-v0', agent_list)
    # env = pommerman.make('PommeFFACompetition-v0', agent_list)

    # Run the episodes just like OpenAI Gym
    for episode in range(30):
        steps_per_episode = 0
        state = env.reset()
        done = False
        while not done:
            # env.render()
            actions = env.act(state)
            new_state, reward, done, info = env.step(actions)
            steps_per_episode += 1
            # agents learning
            for i in range(4):
                dqn_agents[i].remember(
                    state[i], actions[i],
                    reward[i], done, new_state[i]
                )
                dqn_agents[i].experience_replay()

            state = new_state

        print('Episode {} finished\n steps: {}\ninfo: {}'.
            format(episode, steps_per_episode, info))
        # saving weights after every epoch
        for i, dqn_agent in enumerate(dqn_agents):
            dqn_agent.save_weights("./weights/model_{}_episode_{}".format(i, episode))
    env.close()

def create_four_test_models():
    model1 = tf.keras.Sequential([
        tf.keras.layers.Conv2D(8, input_shape=(11, 11, 6),kernel_size=(5, 5), activation='relu', padding='same'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(6, activation='softmax')
    ])
    model1.compile(optimizer=tf.train.AdamOptimizer(0.001), loss='mse', metrics=['mae'])

    model2 = tf.keras.Sequential([
        tf.keras.layers.Conv2D(8, input_shape=(11, 11, 6), kernel_size=(5, 5), activation='relu'),
        tf.keras.layers.Conv2D(16, kernel_size=(5, 5), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(6, activation='softmax')
    ])
    model2.compile(optimizer=tf.train.AdamOptimizer(0.001), loss='mse', metrics=['mae'])

    model3 = tf.keras.Sequential([
        tf.keras.layers.Conv2D(8, input_shape=(11, 11, 6), kernel_size=(5, 5), activation='relu', padding='same'),
        tf.keras.layers.Conv2D(16, input_shape=(11, 11, 6), kernel_size=(5, 5), activation='relu', padding='same'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(6, activation='softmax')
    ])
    model3.compile(optimizer=tf.train.AdamOptimizer(0.001), loss='mse', metrics=['mae'])

    model4 = tf.keras.Sequential([
        tf.keras.layers.Conv2D(8, input_shape=(11, 11, 6), kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.Conv2D(16, input_shape=(11, 11, 6), kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(6, activation='softmax')
    ])
    model4.compile(optimizer=tf.train.AdamOptimizer(0.001), loss='mse', metrics=['mae'])

    return model1, model2, model3, model4

def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(8, input_shape=(11, 11, 6), kernel_size=(5, 5), activation='relu', padding='same'),
        tf.keras.layers.Conv2D(16, input_shape=(11, 11, 6), kernel_size=(5, 5), activation='relu', padding='same'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(6, activation='softmax')
    ])
    model.compile(optimizer=tf.train.AdamOptimizer(0.001), loss='mse', metrics=['mae'])
    return model

if __name__ == '__main__':
    main()
