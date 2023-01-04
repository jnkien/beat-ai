""" Update a RL model """

import argparse
import os

from stable_baselines3 import PPO

import lib.env
import lib.model_rl
import lib.utils


def update_rl(directory: str):
    """Update (continue training) a RL model on an env

    Example:
        python update_rl.py -d="data/models/50bkOHBpXFl2RnGJVImI1MzvI9iXvF26"
    """
    config = lib.utils.load_config(os.path.join(directory, "config.yaml"))
    env = lib.env.create_stacked_env(config["stacks"])

    step_model = lib.utils.get_max_step_rl_model(directory)
    model_path = os.path.join(directory, f"model_{step_model}.zip")
    model = PPO.load(model_path)
    model.set_env(env)

    total_timesteps = config["total_timesteps"]
    freq_to_save = config["freq_to_save"]
    model.learn(
        total_timesteps=total_timesteps,
        callback=lib.model_rl.TrainCallback(
            freq_to_save, directory, n_calls=step_model
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--directory", help="Directory containing the models of one type"
    )
    args = parser.parse_args()

    if args.directory is None:
        raise TypeError("missing 1 positional parameter (-d or --directory)")
    update_rl(args.directory)
