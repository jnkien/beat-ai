""" Update a RL model """

import os

from stable_baselines3 import PPO

import lib.env
import lib.model_rl
import lib.utils


def update_rl():
    """Update (continue training) a RL model on an env"""

    model_dir = "data/models/50bkOHBpXFl2RnGJVImI1MzvI9iXvF26"
    config = lib.utils.load_config(os.path.join(model_dir, "config.yaml"))
    env = lib.env.create_stacked_env(config["stacks"])

    model_path, max_step_model = lib.utils.get_last_rl_model_path(model_dir)
    model = PPO.load(model_path)
    model.set_env(env)

    total_timesteps = config["total_timesteps"]
    freq_to_save = config["freq_to_save"]
    model.learn(
        total_timesteps=total_timesteps,
        callback=lib.model_rl.TrainCallback(
            freq_to_save, model_dir, n_calls=max_step_model
        ),
    )


if __name__ == "__main__":
    update_rl()
