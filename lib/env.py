""" Facades to create environments """

import gym_super_mario_bros
import numpy as np
from gym.wrappers.gray_scale_observation import GrayScaleObservation
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym_super_mario_bros.smb_env import SuperMarioBrosEnv
from nes_py.wrappers import JoypadSpace
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


def create_stacked_env(stacks: int) -> SuperMarioBrosEnv:
    """Create an stacked environement already preprocessed.

    Args:
        stacks: number of stacks for the env

    Returns:
        A SuperMarioBrosEnv object.
    """
    env = create_unstacked_env()
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, stacks, channels_order="last")
    return env


def run(env: SuperMarioBrosEnv, model) -> np.array:
    """Play the environment given model predictions.

    Args:
        env : an environement
        model : a model

    Returns:
        The frames of the run
    """
    states = []
    state = env.reset()
    done = False
    for _ in range(1000):  # limit the simulation
        if done:
            break
        action, _ = model.predict(state)
        state, _, done, _ = env.step(action)
        states.append(state)
        env.render()
    env.close()
    return np.array(states)
