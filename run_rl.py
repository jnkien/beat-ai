""" Let a model play the game """

import argparse
import os
import re

import pandas as pd
from stable_baselines3 import PPO

import lib.env
import lib.utils


def run_rl(directory: str, file: str = None):
    """Run a RL model on an env and export the states as a video

    Example:
        python run_rl.py -d="data/models/50bkOHBpXFl2RnGJVImI1MzvI9iXvF26" -f="model_5000.zip"
    """
    config = lib.utils.load_config(os.path.join(directory, "config.yaml"))

    env = lib.env.create_stacked_env(config["stacks"])
    if file is None:
        step_model = lib.utils.get_max_step_rl_model(directory)
        model_path = os.path.join(directory, f"model_{step_model}.zip")
    else:
        model_path = os.path.join(directory, file)
        regexp = re.compile("^model_([0-9]*).zip$")
        step_model = int(regexp.search(file).group(1))
    model = PPO.load(model_path)

    states, x_pos = lib.env.run(env, model)
    states = states[:, 0, :, :, 3]
    lib.utils.states_to_mp4(states, os.path.join(directory, f"run_rl_{step_model}.mp4"))

    pd.DataFrame(x_pos).to_csv(
        os.path.join(directory, f"x_pos_rl_{step_model}.csv"),
        header=None,
        index=None,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--directory", help="Directory containing the models")
    parser.add_argument("-f", "--file", help="File containing the model")
    args = parser.parse_args()

    if args.directory is None:
        raise TypeError("missing 1 positional parameter (-d or --directory)")
    run_rl(args.directory, args.file)
