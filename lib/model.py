""" Model related functions """

import os

from gym_super_mario_bros.smb_env import SuperMarioBrosEnv
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback


class TrainCallback(BaseCallback):
    """A callback during the training of a RL model."""

    def __init__(self, freq_to_save, save_path, verbose=1) -> None:
        super().__init__(verbose)
        self.freq_to_save = freq_to_save
        self.save_path = save_path

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.freq_to_save == 0:
            model_path = os.path.join(self.save_path, f"model_{self.n_calls}")
            self.model.save(model_path)
        return True


def learn(model_name: str, env: SuperMarioBrosEnv) -> BaseAlgorithm:
    """Learn a RL model on an environment.

    Args:
        model_name : a RL model
        env : an environment

    Returns:
        A learnt RL model
    """
    model = model_factory(model_name, env)
    model.learn(
        total_timesteps=4_000_000, callback=TrainCallback(5_000, "./data/models/")
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
