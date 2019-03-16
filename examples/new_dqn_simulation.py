from pommerman.Qlearning.dqn_keras_rl import create_dqn, set_pommerman_env, create_model, DQN
from pommerman.Qlearning.env_wrapper import EnvWrapper
from pommerman.Qlearning.env_with_rewards import EnvWrapperRS
from pommerman.configs import ffa_v0_fast_env
from pommerman.envs.v0 import Pomme
from pommerman.agents import SimpleAgent, RandomAgent, PlayerAgent

BOARD_SIZE = 11


def env_for_players():
    config = ffa_v0_fast_env()
    env = Pomme(**config["env_kwargs"])
    agents = [RandomAgent(config["agent"](0, config["game_type"])),
              RandomAgent(config["agent"](1, config["game_type"])),
              DQN(config["agent"](2, config["game_type"])),
              PlayerAgent(config["agent"](3, config["game_type"]))]
    env.set_agents(agents)
    env.set_training_agent(agents[2].agent_id)  # training_agent is only dqn agent
    env.set_init_game_state(None)

    return env


def main():
    model = create_model()
    dqn, callbacks = create_dqn(model=model)
<<<<<<< HEAD
    dqn.load_weights('../pommerman/Qlearning/models/15_03_18-40_marcin_weight_with_rewards.h5')
    env = EnvWrapperRS(set_pommerman_env(), BOARD_SIZE)
=======
    dqn.load_weights('../pommerman/Qlearning/models/marcin_weight_14_03_17-20.h5')
    env = EnvWrapper(set_pommerman_env(), BOARD_SIZE)  # change env_for_players() to set_pommerman_env to have a simulation
>>>>>>> parent of 8575ec7... training weight with rewards 16.03
    while True:
        dqn.test(env)


if __name__ == '__main__':
    main()
