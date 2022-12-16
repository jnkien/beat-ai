""" Let a model play the game """

from stable_baselines3 import PPO

import lib.env


def run():
    """Run a model on an env"""
    env = lib.env.create_stacked_env()
    model = PPO.load("ppo_mario.zip")
    lib.env.run(env, model)


if __name__ == "__main__":
    run()
