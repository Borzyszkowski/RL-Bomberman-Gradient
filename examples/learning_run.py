import pommerman
from pommerman import agents
import tensorflow as tf

def main():
    # Create agent able to learn
    model = create_model()
    target_model = create_model()
    dqn_agent = agents.DQN_agent(model, target_model, 
        epsilon=0.5, ep_decay=0.995, batch_size=128, memory_length=1000)
    dqn_agent_index = 0

    # Create a set of two agents
    agent_list = [
        dqn_agent,
        agents.SimpleAgent()
    ]

    # Make the "Free-For-All" environment using the agent list
    env = pommerman.make('PommeFFACompetitionFast-v0', agent_list)
    # env = pommerman.make('PommeFFACompetition-v0', agent_list)

    # Run the episodes just like OpenAI Gym
    for episode in range(100):
        steps_per_episode = 0
        state = env.reset()
        done = False
        while not done:
            # env.render()
            actions = env.act(state)
            new_state, reward, done, info = env.step(actions)
            steps_per_episode += 1
            # agents learning
            dqn_agent.remember(
                state[dqn_agent_index], actions[dqn_agent_index],
                reward[dqn_agent_index], done, new_state[dqn_agent_index]
            )
            
            state = new_state

        dqn_agent.experience_replay()
        dqn_agent.target_train()

        print('Episode {} finished, steps: {}\ninfo: {}'.
            format(episode, steps_per_episode, info))
        if episode % 10 == 9:
            # saving weights after every epoch
            dqn_agent.save_weights("./weights/best_model_episode_{}".format(episode))
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
    model3.compile(optimizer=tf.train.AdamOptimizer(0.001), loss='categorical_crossentropy', metrics=['mae'])

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
    model.compile(optimizer=tf.train.AdamOptimizer(0.001), loss='categorical_crossentropy', metrics=['mae'])
    return model

if __name__ == '__main__':
    main()
