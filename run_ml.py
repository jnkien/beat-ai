""" Let a model play the game """

import os

import gym_super_mario_bros.actions
import torch

import lib.env
import lib.model_ml
import lib.utils


def run_ml():
    """Run a ML model on an env and export the states as a video"""
    model_dir = "data/models/b6YCgIyFr1sO5iWtTSKkdvEH4Smm8Lgr"
    config = lib.utils.load_config(os.path.join(model_dir, "config.yaml"))

    env = lib.env.create_stacked_env(config["stacks"])
    model = lib.model_ml.CNNModel(len(gym_super_mario_bros.actions.SIMPLE_MOVEMENT), 1)
    model.load_state_dict(torch.load(os.path.join(model_dir, "model_cnn.pyt")))

    states, _ = lib.env.run(env, model)
    states = states[:, 0, :, :, 3]
    lib.utils.states_to_mp4(states, os.path.join(model_dir, "run.mp4"))


if __name__ == "__main__":
    run_ml()
