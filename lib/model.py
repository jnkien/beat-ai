""" Model related functions """

import os
import random
import string

from gym_super_mario_bros.smb_env import SuperMarioBrosEnv
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback

import lib.utils


class TrainCallback(BaseCallback):
    """A callback during the training of a RL model."""

    def __init__(
        self, freq_to_save: int, save_path: str, n_calls: int = 0, verbose=1
    ) -> None:
        super().__init__(verbose)
        self.freq_to_save = freq_to_save
        self.save_path = save_path
        self.n_calls = n_calls

    def _on_step(self) -> bool:
        if self.n_calls % self.freq_to_save == 0:
            model_path = os.path.join(self.save_path, f"model_{self.n_calls}")
            self.model.save(model_path)
        return True


def learn(config: dict, env: SuperMarioBrosEnv) -> BaseAlgorithm:
    """Learn a RL model on an environment.

    Args:
        config : config parameters as a dict
        env : an environment

    Returns:
        A learnt RL model
    """
    model_name = config["model"]["name"]
    total_timesteps = config["model"]["total_timesteps"]
    freq_to_save = config["model"]["freq_to_save"]
    save_path = config["model"]["save_path"]

    model = model_factory(model_name, env)

    dir_hash = "".join(random.choices(string.ascii_letters + string.digits, k=32))
    save_path = os.path.join(save_path, dir_hash)
    os.makedirs(save_path, exist_ok=False)

    lib.utils.dump_config(config, os.path.join(save_path, "config.yaml"))

    model.learn(
        total_timesteps=total_timesteps, callback=TrainCallback(freq_to_save, save_path)
    )


def model_factory(model_name: str, env: SuperMarioBrosEnv) -> BaseAlgorithm:
    """A factory of RL models.

    Args:
        model_name : name of the model
        env : environement for initializing the model

    Returns:
        The initialized model.
    """
    factory = {
        "PPO": PPO(
            "CnnPolicy",
            env,
            verbose=1,
            tensorboard_log="./logs/",
            learning_rate=0.000001,
            n_steps=512,
        )
    }
    return factory[model_name]
