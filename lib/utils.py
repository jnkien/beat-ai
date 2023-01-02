""" Utils functions """

import numpy as np
import yaml
from matplotlib import pyplot as plt


def load_config(path: str) -> dict:
    """Load the config parameters from a yaml file

    Args:
        path : path of the yaml file

    Returns:
        Config parameters
    """
    with open(path, encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.BaseLoader)
    config["model"]["stacks"] = int(config["model"]["stacks"])
    config["model"]["total_timesteps"] = int(config["model"]["total_timesteps"])
    config["model"]["freq_to_save"] = int(config["model"]["freq_to_save"])
    return config


def dump_config(config: dict, path: str) -> None:
    """Dump the config parameters o a yaml file

    Args:
        config : the config parameters
        path : path of the yaml file
    """
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config, f)


def save_frame_seq(state: np.array, path: str) -> None:
    """Save a frame from a stacked environement in png.

    Args:
        state : a frame of a stacked environment
        path : path of the file
    """
    plt.figure(figsize=(20, 16))
    for idx in range(state.shape[3]):
        plt.subplot(1, 4, idx + 1)
        plt.imshow(state[0][:, :, idx])
    plt.savefig(path)


def save_frame(state: np.array, path: str) -> None:
    """Save a frame from an environement in png.

    Args:
        state : a frame of an environment
        path : path of the file
    """
    plt.figure()
    plt.imshow(state)
    plt.savefig(path)
    plt.close()
