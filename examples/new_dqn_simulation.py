from pommerman.Qlearning.dqn_keras_rl import load_trained_model, get_dqn, get_env, create_model
from pommerman.Qlearning.evaluation_utils import run_episode
from pommerman.Qlearning.env_wrapper import EnvWrapper
from pommerman.configs import ffa_v0_fast_env

BOARD_SIZE = 11


def main():
    model = create_model()
    dqn, callbacks = get_dqn(model=model)
    dqn.load_weights('../pommerman/Qlearning/models/dqn_agent_full_model.hdf5')
    config = ffa_v0_fast_env()
    env = EnvWrapper(get_env(), BOARD_SIZE)
    # info, reward, lens = run_episode(dqn, config, env=env)
    dqn.test(env)


if __name__ == '__main__':
    main()
