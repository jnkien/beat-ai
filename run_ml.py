""" Let a model play the game """

import argparse
import os

import gym_super_mario_bros.actions
import torch

import lib.env
import lib.model_ml
import lib.utils


def run_ml(directory: str):
    """Run a ML model on an env and export the states as a video

    Example:
        python run_ml.py -d="data/models/b6YCgIyFr1sO5iWtTSKkdvEH4Smm8Lgr"
    """
    config = lib.utils.load_config(os.path.join(directory, "config.yaml"))

    env = lib.env.create_stacked_env(config["stacks"])
    model = lib.model_ml.CNNModel(len(gym_super_mario_bros.actions.SIMPLE_MOVEMENT), 1)
    model.load_state_dict(torch.load(os.path.join(directory, "model_cnn.pyt")))

    states, _ = lib.env.run(env, model)
    states = states[:, 0, :, :, 3]
    lib.utils.states_to_mp4(states, os.path.join(directory, "run_ml.mp4"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--directory", help="Directory containing the ML model")
    args = parser.parse_args()

    if args.directory is None:
        raise TypeError("missing 1 positional parameter (-d or --directory)")
    run_ml(args.directory)
