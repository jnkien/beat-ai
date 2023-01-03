""" Train a RL model """

import lib.env
import lib.model_rl
import lib.utils


def train_rl():
    """Train a RL model on an env"""

    config = lib.utils.load_config("config_rl.yaml")
    env = lib.env.create_stacked_env(config["stacks"])
    lib.model_rl.learn(config, env)


if __name__ == "__main__":
    train_rl()
