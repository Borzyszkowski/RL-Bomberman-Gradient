from pommerman.Qlearning.dqn_keras_rl import create_dqn, set_pommerman_env, create_model, DQN
from pommerman.Qlearning.env_wrapper import EnvWrapper
from pommerman.Qlearning.env_with_rewards import EnvWrapperRS
from pommerman.configs import ffa_v0_fast_env
from pommerman.envs.v0 import Pomme
from pommerman.agents import SimpleAgent, RandomAgent, PlayerAgent
import time

BOARD_SIZE = 11


def env_for_players():
    config = ffa_v0_fast_env(50)
    env = Pomme(**config["env_kwargs"])
    agents = [DQN(config["agent"](0, config["game_type"])),
              PlayerAgent(config["agent"](1, config["game_type"])),
             RandomAgent(config["agent"](2, config["game_type"])),
             RandomAgent(config["agent"](2, config["game_type"]))]
    env.set_agents(agents)
    env.set_training_agent(agents[0].agent_id)  # training_agent is only dqn agent
    env.set_init_game_state(None)

    return env

def main():
    model = create_model()
    dqn, callbacks = create_dqn(model=model)
    dqn.load_weights('../pommerman/Qlearning/models/18_03_9-10_new_reward.h5')
    while True:
        input("Click enter to play...")
        env = EnvWrapperRS(env_for_players(), BOARD_SIZE)  # change env_for_players() to set_pommerman_env to have a simulation
        dqn.test(env)

if __name__ == '__main__':
    main()
