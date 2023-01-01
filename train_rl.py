""" Train a RL model """

import yaml

import lib.env
import lib.model


def train_rl():
    """Train a RL model on an env"""

    with open("config.yaml", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.BaseLoader)

    env = lib.env.create_stacked_env()
    lib.model.learn(config["model"][0]["name"], env)


if __name__ == "__main__":
    train_rl()
