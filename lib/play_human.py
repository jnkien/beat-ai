"""A method to play gym environments using human IO inputs."""
import os
import random
import string
import time
from dataclasses import dataclass

import gym
import numpy as np
from nes_py._image_viewer import ImageViewer
from pyglet import clock


@dataclass
class SaveCallback:
    """A callback for saving the actions and states of an human run."""

    save_path: str

    def __post_init__(self) -> None:
        """After __init__() tasks."""
        run_hash = "".join(random.choices(string.ascii_letters + string.digits, k=32))
        self.save_path = os.path.join(self.save_path, run_hash)
        os.makedirs(self.save_path, exist_ok=False)

    def _save_actions(self, actions: np.array) -> None:
        """Save the actions in a txt file.

        Args:
            actions : actions done during all the steps
        """
        np.savetxt(
            os.path.join(self.save_path, "actions.csv"),
            [int(action) for action in actions],
        )

    def _save_states(self, states: np.array) -> None:
        """Save the states in a .npy file

        Args:
            states : states produced during all the steps
        """
        np.save(os.path.join(self.save_path, "states.npy"), states, allow_pickle=True)

    def call(self, states: np.array, actions: np.array) -> None:
        """Save both actions and states.

        Args:
            states : states produced during all the steps
            actions : actions done during all the steps
        """
        self._save_states(states)
        self._save_actions(actions)


# the sentinel value for "No Operation"
_NOP = 0


def play_human(env: gym.Env, callback=None) -> None:  # pylint: disable=R0914
    """
    Play the environment using keyboard as a human.

    Args:
        env: the initialized gym environment to play
        callback: a callback to receive output from the environment

    Returns:
        None

    """
    # ensure the observation space is a box of pixels
    assert isinstance(env.observation_space, gym.spaces.box.Box)
    # ensure the observation space is either B&W pixels or RGB Pixels
    obs_s = env.observation_space
    is_bw = len(obs_s.shape) == 2
    is_rgb = len(obs_s.shape) == 3 and obs_s.shape[2] in [1, 3]
    assert is_bw or is_rgb
    # get the mapping of keyboard keys to actions in the environment
    if hasattr(env, "get_keys_to_action"):
        keys_to_action = env.get_keys_to_action()
    elif hasattr(env.unwrapped, "get_keys_to_action"):
        keys_to_action = env.unwrapped.get_keys_to_action()
    else:
        raise ValueError("env has no get_keys_to_action method")
    # create the image viewer
    viewer = ImageViewer(
        env.spec.id if env.spec is not None else env.__class__.__name__,
        env.observation_space.shape[0],  # height
        env.observation_space.shape[1],  # width
        monitor_keyboard=True,
        relevant_keys=set(sum(map(list, keys_to_action.keys()), [])),
    )
    # create a done flag for the environment
    done = True
    # prepare frame rate limiting
    target_frame_duration = 1 / env.metadata["video.frames_per_second"]
    last_frame_time = 0
    # Init storing objects
    actions = []
    states = []
    # start the main game loop
    try:
        while True:
            current_frame_time = time.time()
            # limit frame rate
            if last_frame_time + target_frame_duration > current_frame_time:
                continue
            # save frame beginning time for next refresh
            last_frame_time = current_frame_time
            # clock tick
            clock.tick()
            # reset if the environment is done
            if done:
                done = False
                state = env.reset()
                viewer.show(env.unwrapped.screen)
            # unwrap the action based on pressed relevant keys
            action = keys_to_action.get(viewer.pressed_keys, _NOP)
            actions.append(action)
            states.append(state)
            next_state, _, done, _ = env.step(action)
            viewer.show(env.unwrapped.screen)
            state = next_state
            # shutdown if the escape key is pressed
            if viewer.is_escape_pressed:
                break
        if callback is not None:
            callback(states, actions)
    except KeyboardInterrupt:
        pass

    viewer.close()
    env.close()


# explicitly define the outward facing API of the module
__all__ = [play_human.__name__]
