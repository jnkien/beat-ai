""" Let a model play the game """

import os

from stable_baselines3 import PPO

import lib.env
import lib.utils


def run_rl():
    """Run a RL model on an env and export the states as a video"""
    model_dir = "data/models/50bkOHBpXFl2RnGJVImI1MzvI9iXvF26"
    config = lib.utils.load_config(os.path.join(model_dir, "config.yaml"))

    env = lib.env.create_stacked_env(config["stacks"])
    model_path, _ = lib.utils.get_last_rl_model_path(model_dir)
    model = PPO.load(model_path)

    states = lib.env.run(env, model)
    states = states[:, 0, :, :, 3]
    lib.utils.states_to_mp4(states, os.path.join(model_dir, "run.mp4"))


if __name__ == "__main__":
    run_rl()
