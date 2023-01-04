""" Utils functions """

import functools
import os
import re
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import yaml
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split


def load_config(path: str) -> dict:
    """Load the config parameters from a yaml file

    Args:
        path : path of the yaml file

    Returns:
        Config parameters
    """
    with open(path, encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.BaseLoader)

    config = conf_param_to_int(
        config,
        ["stacks", "seed", "total_timesteps", "freq_to_save", "batch_size", "n_iters"],
    )
    config = conf_param_to_float(config, ["learning_rate", "test_size"])
    return config


def conf_param_to_int(config: dict, keys: List[str]) -> dict:
    """Cast config parameters as int

    Args:
        config : the config parameters
        keys : parameters to cast as int

    Returns:
        The config parameters with the desired ones cast as int
    """
    for key in keys:
        if key in config:
            config[key] = int(config[key])
    return config


def conf_param_to_float(config: dict, keys: List[str]) -> dict:
    """Cast config parameters as float

    Args:
        config : the config parameters
        keys : parameters to cast as float

    Returns:
        The config parameters with the desired ones cast as float
    """
    for key in keys:
        if key in config:
            config[key] = float(config[key])
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


def npy_to_mp4(npy_path: str, out_path: str) -> None:
    """Convert a sequence of frames (from .npy) as a video

    Args:
        npy_path : path of the .npy file
        out_path : path to export
    """
    states = np.load(npy_path)
    states_to_mp4(states, out_path)


def states_to_mp4(states: np.array, out_path: str):
    """Convert a sequence of frames (from np.array) as a video

    Args:
        states : a sequence of frames
        out_path : path to export
    """
    size = (states.shape[1], states.shape[2])
    fps = 60
    out = cv2.VideoWriter(  # pylint: disable=E1101
        out_path,
        cv2.VideoWriter_fourcc(*"mp4v"),  # pylint: disable=E1101
        fps,
        (size[1], size[0]),
        False,
    )
    for state in states:
        out.write(state)
    out.release()


def load_human_data(data_path: str) -> Tuple[np.array, List[np.array]]:
    """Load all human data generated by generate_human_data.py

    Args:
        data_path : Path of the directory containing all the human data

    Returns:
        Actions and States generated by humans
    """
    actions = []
    states = []
    for d in os.listdir(data_path):
        actions_filepath = os.path.join(data_path, d, "actions.csv")
        actions_i = pd.read_csv(actions_filepath, header=None)
        print(f"[INFO] File {actions_filepath} contains {actions_i.shape[0]} actions")
        actions.append(actions_i)

        states_filepath = os.path.join(data_path, d, "states.npy")
        states_i = np.load(states_filepath)
        print(
            f"[INFO] File {states_filepath} contains {states_i.shape[0]} states of shape \
                ({states_i.shape[1]}, {states_i.shape[2]})"
        )
        states.append(np.load(states_filepath))

    actions = pd.concat(actions)
    states = functools.reduce(lambda a, b: np.concatenate((a, b), axis=0), states)

    print(f"[INFO] {actions.shape[0]} actions collected")
    print(
        f"[INFO] {states.shape[0]} states collected of shape ({states.shape[1]}, {states.shape[2]})"
    )

    actions = np.array(actions[0])
    return actions, states


def transform_data(  # pylint: disable=R0914
    actions: np.array, states: List[np.array], config: dict
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Transform data for model training

    Args:
        actions : a list of input command
        states : a list of frames
        config : a config file

    Returns:
        Train and test data ready to feed a model
    """
    stacks = config["stacks"]
    seed = config["seed"]
    batch_size = config["batch_size"]
    test_size = config["test_size"]

    X = states.reshape(
        states.shape[0], states.shape[1], states.shape[2]
    )  # drop last dimension for vectorizing

    # vectorize the frame by stacks
    u = np.lib.stride_tricks.sliding_window_view(np.arange(X.shape[0]), stacks)
    u = u.flatten()
    X = X[u].reshape(-1, X.shape[1], X.shape[2], stacks)
    X = X.reshape(
        X.shape[0], X.shape[1], X.shape[2], X.shape[3], 1
    )  # add one channel because frames are b&w

    # truncate the start (time to get enough frames to associate an action to a stack of frame)
    actions = actions[stacks - 1 :]

    X_train, X_test, target_train, target_test = train_test_split(
        X, actions, test_size=test_size, random_state=seed
    )

    train_x = (
        torch.from_numpy(X_train)  # pylint: disable=E1101
        .permute(0, 4, 3, 1, 2)
        .float()
    )
    train_y = torch.from_numpy(target_train).long()  # pylint: disable=E1101
    test_x = (
        torch.from_numpy(X_test).permute(0, 4, 3, 1, 2).float()  # pylint: disable=E1101
    )
    test_y = torch.from_numpy(target_test).long()  # pylint: disable=E1101

    train = torch.utils.data.TensorDataset(train_x, train_y)
    test = torch.utils.data.TensorDataset(test_x, test_y)

    train_loader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        test, batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader


def get_max_step_rl_model(model_dir: str) -> str:
    """Get the last RL model path

    Args:
        model_dir : the directory to search in

    Returns:
        The max step available for the model
    """
    regexp = re.compile("^model_([0-9]*).zip$")
    return max(
        int(regexp.search(x).group(1))
        for x in os.listdir(model_dir)
        if regexp.search(x)
    )


def generate_x_pos_fig(path: str, out_path: str) -> None:
    """Generate the figure of the distance travelled by Mario

    Example:
        generate_x_pos_fig("data/models/50bkOHBpXFl2RnGJVImI1MzvI9iXvF26", 'img/x_pos.png')

    Args:
        path : path of the x_pos files
        out_path : path of the figure
    """
    regexp = re.compile("x_pos_rl_([0-9]*).csv")

    x_pos = {}
    for file in [f for f in os.listdir(path) if ".csv" in f]:
        x_pos_i = pd.read_csv(os.path.join(path, file), header=None)
        x_pos[f"{regexp.search(file).group(1)} frames"] = np.array(x_pos_i[0])

    for k, item in x_pos.items():
        plt.plot(np.arange(item.shape[0]), item, label=k)
    plt.legend()
    plt.savefig(out_path)
    plt.close()
