""" Facades to create environments """

import gym_super_mario_bros
from gym.wrappers.gray_scale_observation import GrayScaleObservation
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym_super_mario_bros.smb_env import SuperMarioBrosEnv
from nes_py.wrappers import JoypadSpace
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.vec_frame_stack import VecFrameStack


def create_unstacked_env() -> SuperMarioBrosEnv:
    """Create an unstacked environement already preprocessed.

    Returns:
        A SuperMarioBrosEnv object.
    """
    env = gym_super_mario_bros.make("SuperMarioBros-v0")
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = GrayScaleObservation(env, keep_dim=True)
    return env


def create_stacked_env() -> SuperMarioBrosEnv:
    """Create an stacked environement already preprocessed.

    Returns:
        A SuperMarioBrosEnv object.
    """
    env = create_unstacked_env()
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, 4, channels_order="last")
    return env


def run(env: SuperMarioBrosEnv, model: BaseAlgorithm) -> None:
    """Play the environment given model predictions.

    Args:
        env : an environement
        model : a model
    """
    state = env.reset()
    done = False
    while not done:
        action, _ = model.predict(state)
        state, _, done, _ = env.step(action)
        env.render()
