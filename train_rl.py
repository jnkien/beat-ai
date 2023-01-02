""" Train a RL model """

import lib.env
import lib.model
import lib.utils


def train_rl():
    """Train a RL model on an env"""

    config = lib.utils.load_config("config.yaml")
    env = lib.env.create_stacked_env(config["model"]["stacks"])
    lib.model.learn(config, env)


if __name__ == "__main__":
    train_rl()
