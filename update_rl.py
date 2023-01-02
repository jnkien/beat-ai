""" Update a RL model """

import os
import re

from stable_baselines3 import PPO

import lib.env
import lib.model
import lib.utils


def update_rl():
    """Update (continue training) a RL model on an env"""

    model_dir = "data/models/50bkOHBpXFl2RnGJVImI1MzvI9iXvF26"
    config = lib.utils.load_config(os.path.join(model_dir, "config.yaml"))
    env = lib.env.create_stacked_env(config["model"]["stacks"])

    regexp = re.compile("^model_([0-9]*).zip$")
    max_step_model = max(
        int(regexp.search(x).group(1))
        for x in os.listdir(model_dir)
        if regexp.search(x)
    )

    model = PPO.load(os.path.join(model_dir, f"model_{max_step_model}.zip"))
    model.set_env(env)

    total_timesteps = config["model"]["total_timesteps"]
    freq_to_save = config["model"]["freq_to_save"]
    model.learn(
        total_timesteps=total_timesteps,
        callback=lib.model.TrainCallback(
            freq_to_save, model_dir, n_calls=max_step_model
        ),
    )


if __name__ == "__main__":
    update_rl()
